"""
groq_client.py
--------------
Everything that talks to the Groq API lives in this one file.

Groq hosts open-weight LLMs (like OpenAI's gpt-oss) and exposes an
"OpenAI-compatible" API: we send a list of chat messages and get the model's
reply back as JSON. We use the plain `requests` library so it's easy to see
exactly what goes over the network.

Reliability features (important for a live demo!):
  * The API key is read from the .env file only - it never reaches the browser.
  * HTTP 429 "rate limit" -> wait (using Groq's retry-after header, or 3 s)
    and try again, up to 3 times.
  * If the main model is still rate-limited (or unavailable), we try the
    smaller fallback model once, so the show can go on.
  * gpt-oss models are REASONING models: they "think" before answering, and
    that thinking uses up max_tokens too. We ask for low reasoning effort,
    give generous max_tokens, read ONLY message.content (never the separate
    reasoning field), and retry once if the answer comes back empty.
  * Every failure becomes a GroqError with a friendly message that the
    Flask app turns into clean JSON for the frontend.
"""

import os
import time

import requests
from dotenv import load_dotenv

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-20b"
REASONING_EFFORT = "low"       # less hidden "thinking" = faster, cheaper answers
DEFAULT_MAX_TOKENS = 1500      # covers the reasoning AND the final answer

MAX_RATE_LIMIT_RETRIES = 3     # how many times we retry after a 429
DEFAULT_RETRY_WAIT = 3.0       # seconds to wait if Groq doesn't tell us
MAX_RETRY_WAIT = 15.0          # never freeze the demo for too long
REQUEST_TIMEOUT = 45           # seconds before we give up on one request

# The .env file sits next to this script, so we load it by absolute path.
# That way the app works no matter which folder you launch it from.
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


class GroqError(Exception):
    """A friendly, user-facing error.

    status       -> HTTP status code the Flask route should return
    fallback_ok  -> True if trying the fallback model might help
    json_failed  -> True if the model failed to produce valid JSON
    """

    def __init__(self, message, status=502, fallback_ok=False, json_failed=False):
        super().__init__(message)
        self.message = message
        self.status = status
        self.fallback_ok = fallback_ok
        self.json_failed = json_failed


def load_settings():
    """Read the API key and model names from .env.

    We re-read the file on every call, so if you paste your key into .env
    while the server is running, it starts working without a restart.
    """
    load_dotenv(ENV_PATH, override=True)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key in ("", "paste_your_key_here"):
        api_key = ""  # treat the placeholder as "no key"
    model = os.getenv("GROQ_MODEL", "").strip() or DEFAULT_MODEL
    fallback = os.getenv("GROQ_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL).strip()
    return {"api_key": api_key, "model": model, "fallback_model": fallback}


def chat(messages, json_mode=False, temperature=0.7, max_tokens=DEFAULT_MAX_TOKENS):
    """Send chat `messages` to Groq and return the reply text (a string).

    messages    -> [{"role": "system" | "user" | "assistant", "content": "..."}]
    json_mode   -> ask Groq to force valid JSON output (used by the judge)
    temperature -> creativity: ~0.2 = consistent judge, ~0.9 = lively debater
    """
    settings = load_settings()
    if not settings["api_key"]:
        raise GroqError("API key missing: add GROQ_API_KEY to your .env file", status=500)

    args = (settings["api_key"], messages, json_mode, temperature, max_tokens)
    try:
        return _answer(settings["model"], *args)
    except GroqError as err:
        fallback = settings["fallback_model"]
        # Plan B: if the big model is overloaded or unavailable, use the small one.
        if err.fallback_ok and fallback and fallback != settings["model"]:
            print(f"[groq] {err.message} -> switching to fallback model {fallback}")
            return _answer(fallback, *args)
        raise


def _answer(model, *args):
    """Call one model; if its answer is empty, retry once before giving up."""
    for attempt in range(2):
        text = _call_model(model, *args)
        if text.strip():
            return text
        print(f"[groq] empty answer from {model} (attempt {attempt + 1}/2)")
    raise GroqError("The AI sent back an empty answer. Please try again.", status=502)


def _call_model(model, api_key, messages, json_mode, temperature, max_tokens):
    """POST one request to Groq, retrying on rate limits / server hiccups."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": REASONING_EFFORT,
    }
    if json_mode:
        # "Structured output": Groq guarantees the reply is a JSON object.
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload,
                                 timeout=REQUEST_TIMEOUT)
        except requests.exceptions.Timeout:
            raise GroqError("Groq took too long to answer. Please try again.", status=504)
        except requests.exceptions.ConnectionError:
            raise GroqError("Could not reach Groq. Check your internet connection.", status=503)

        # --- Rate limit (429) or temporary Groq outage (5xx): wait and retry ---
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < MAX_RATE_LIMIT_RETRIES:
                wait = _retry_wait_seconds(resp)
                _log_retry(resp, model, wait, attempt)
                time.sleep(wait)
                continue
            print(f"[groq] HTTP {resp.status_code} on {model}: giving up after "
                  f"{MAX_RATE_LIMIT_RETRIES} retries", flush=True)
            if resp.status_code == 429:
                raise GroqError("Groq rate limit reached. Wait a few seconds and try again.",
                                status=429, fallback_ok=True)
            raise GroqError("Groq is having trouble right now. Please try again.",
                            status=502, fallback_ok=True)

        if resp.status_code == 401:
            raise GroqError("Your Groq API key was rejected. Check GROQ_API_KEY in your .env file.",
                            status=401)

        if not resp.ok:
            detail = _error_detail(resp)
            # Model name typo or model retired -> the fallback model may still work.
            model_problem = "model" in detail.lower() and resp.status_code in (400, 404)
            # In JSON mode Groq returns 400 if the model wrote invalid JSON.
            json_problem = "json" in detail.lower() and resp.status_code == 400
            raise GroqError(f"Groq error ({resp.status_code}): {detail}",
                            status=502, fallback_ok=model_problem, json_failed=json_problem)

        try:
            choice = resp.json()["choices"][0]
            # Only the final answer. The model's "reasoning" field is ignored.
            content = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise GroqError("Groq sent back a response we could not read. Please try again.")
        if not content.strip():
            print(f"[groq] {model} returned no content (finish_reason={choice.get('finish_reason')})")
        return content

    # The loop always returns or raises, but keep Python happy.
    raise GroqError("Groq request failed. Please try again.")


def _log_retry(resp, model, wait, attempt):
    """Print WHY Groq made us wait, so a slow debate is easy to diagnose.

    For a 429, Groq's message says which limit was hit (requests or tokens
    per minute/day), and its x-ratelimit-* headers show what is left.
    """
    kind = "RATE LIMIT (429)" if resp.status_code == 429 else f"HTTP {resp.status_code}"
    print(f"[groq] {kind} on {model}: waiting {wait:.1f}s, then retry "
          f"{attempt + 1}/{MAX_RATE_LIMIT_RETRIES}", flush=True)
    if resp.status_code == 429:
        h = resp.headers
        print(f"[groq]   Groq says: {_error_detail(resp)}", flush=True)
        print(f"[groq]   left this minute: {h.get('x-ratelimit-remaining-tokens', '?')} tokens "
              f"(resets in {h.get('x-ratelimit-reset-tokens', '?')}), "
              f"{h.get('x-ratelimit-remaining-requests', '?')} requests "
              f"(resets in {h.get('x-ratelimit-reset-requests', '?')})", flush=True)


def _retry_wait_seconds(resp):
    """How long to wait before retrying: Groq's retry-after header, else 3 s."""
    header = resp.headers.get("retry-after")
    try:
        wait = float(header) if header else DEFAULT_RETRY_WAIT
    except ValueError:
        wait = DEFAULT_RETRY_WAIT
    return max(0.5, min(wait, MAX_RETRY_WAIT))


def _error_detail(resp):
    """Pull the human-readable message out of a Groq error response."""
    try:
        return resp.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return resp.text[:200] or "unknown error"
