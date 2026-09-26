"""
app.py - AI Debate Arena (Flask backend)
----------------------------------------
Run it with:   python app.py   then open http://127.0.0.1:5000

The browser never talks to Groq directly. It calls OUR small API below,
and this server adds the secret API key and the prompts. Routes:

  GET  /                 -> the web page (public/index.html)
  GET  /api/status       -> which model is used and whether the key is set
  POST /api/speech       -> one debater speech   (Nova or Rex)
  POST /api/judge-round  -> Judge Sage's scores for one round
  POST /api/verdict      -> the final verdict (winner computed here in Python)
"""

import json
import os
import re

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

import prompts
from groq_client import GroqError, chat, load_settings

# The frontend lives in the "public" folder so the SAME files work both here
# (local Flask) and on Netlify. static_url_path="" serves public/css/style.css
# at /css/style.css, exactly like Netlify does.
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # always serve fresh CSS/JS while developing

# Input limits (validated on every request - never trust the browser)
TOPIC_MIN, TOPIC_MAX = 3, 200
ROUNDS_MIN, ROUNDS_MAX = 2, 4
SPEECH_MAX_CHARS = 3000
CRITERIA = ("logic", "evidence", "rebuttal")


class ApiError(Exception):
    """Raised when the request data is invalid -> returned as a 400 JSON error."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


# ---------------------------------------------------------------------------
# Error handlers: every problem becomes {"error": "..."} instead of a crash
# ---------------------------------------------------------------------------

@app.errorhandler(ApiError)
def handle_api_error(err):
    return jsonify(error=err.message), err.status


@app.errorhandler(GroqError)
def handle_groq_error(err):
    return jsonify(error=err.message), err.status


@app.errorhandler(Exception)
def handle_unexpected(err):
    if isinstance(err, HTTPException):  # e.g. 404 Not Found
        return jsonify(error=err.description), err.code
    app.logger.exception("Unexpected server error")
    return jsonify(error=f"Something went wrong on the server: {err}"), 500


# ---------------------------------------------------------------------------
# Pages and routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.get("/api/status")
def status():
    settings = load_settings()
    return jsonify(
        model=settings["model"],
        fallback_model=settings["fallback_model"],
        key_configured=bool(settings["api_key"]),  # never send the key itself!
    )


@app.post("/api/speech")
def speech():
    """Generate one debater speech. Body: {topic, side, round, total_rounds, history}"""
    body = get_body()
    topic = validate_topic(body.get("topic"))
    side = validate_side(body.get("side"))
    total_rounds = validate_int(body.get("total_rounds"), "total_rounds", ROUNDS_MIN, ROUNDS_MAX)
    round_no = validate_int(body.get("round"), "round", 1, total_rounds)
    history = validate_history(body.get("history"))
    names, _ = get_names(body)

    messages = prompts.build_debater_messages(topic, side, round_no, total_rounds, history, names)
    # Higher temperature = more lively, varied speeches.
    # max_tokens covers the model's hidden reasoning too, so it is much larger
    # than the 70-110 word speech itself (clean_speech trims any extra length).
    raw = chat(messages, temperature=0.85, max_tokens=1500)
    text = clean_speech(raw, prompts.DEBATERS[side]["name"])
    if not text:
        raise ApiError("The debater went quiet (empty reply). Please try again.", 502)
    return jsonify(text=text, side=side, speaker=prompts.DEBATERS[side]["name"])


@app.post("/api/judge-round")
def judge_round():
    """Score one round. Body: {topic, round, for_text, against_text}"""
    body = get_body()
    topic = validate_topic(body.get("topic"))
    total_rounds = validate_int(body.get("total_rounds", ROUNDS_MAX), "total_rounds",
                                ROUNDS_MIN, ROUNDS_MAX)
    round_no = validate_int(body.get("round"), "round", 1, total_rounds)
    for_text = validate_text(body.get("for_text"), "for_text")
    against_text = validate_text(body.get("against_text"), "against_text")
    names, _ = get_names(body)

    messages = prompts.build_judge_round_messages(topic, round_no, total_rounds,
                                                  for_text, against_text, names)
    # Low temperature = consistent, fair scoring.
    data = ask_for_json(messages, required=("for", "against"), temperature=0.2,
                        max_tokens=1500)

    result = {"comment": clean_line(data.get("comment")) or "A close round on both sides."}
    for side in ("for", "against"):
        scores = data.get(side) if isinstance(data.get(side), dict) else {}
        clean = {c: clamp_score(scores.get(c)) for c in CRITERIA}
        clean["total"] = sum(clean[c] for c in CRITERIA)
        result[side] = clean
    return jsonify(result)


@app.post("/api/verdict")
def verdict():
    """Final verdict. Body: {topic, history, totals, mode}"""
    body = get_body()
    topic = validate_topic(body.get("topic"))
    history = validate_history(body.get("history"))
    if not history:
        raise ApiError("The debate has no speeches yet, so there is nothing to judge.")
    names, human_side = get_names(body)

    totals_in = body.get("totals")
    if not isinstance(totals_in, dict):
        raise ApiError("totals must be an object like {\"for\": 20, \"against\": 18}.")
    totals = {s: validate_int(totals_in.get(s), f"totals.{s}", 0, 1000) for s in ("for", "against")}

    # THE MATHS IS DONE IN PYTHON, not by the AI: higher total wins.
    if totals["for"] > totals["against"]:
        winner = "for"
    elif totals["against"] > totals["for"]:
        winner = "against"
    else:
        winner = "draw"

    required = ("reason", "best_line", "tip")
    if human_side:
        required += ("your_strengths", "improve")
    messages = prompts.build_verdict_messages(topic, history, names, totals, winner, human_side)

    try:
        data = ask_for_json(messages, required=required, temperature=0.4, max_tokens=2000,
                            partial_ok=True)
    except GroqError as err:
        if not err.json_failed:
            raise
        data = {}  # the winner is already known, so fall back to safe defaults below

    result = {
        "winner": winner,
        "winner_name": names[winner] if winner != "draw" else None,
        "totals": totals,
        "reason": clean_line(data.get("reason")) or default_reason(winner, names, totals),
        "best_line": clean_line(data.get("best_line")).strip('"“”') or pick_best_line(history, winner),
        "tip": clean_line(data.get("tip")) or "Answer your opponent's strongest point first, then back every claim with a concrete example.",
    }
    # Who said the best line? Search the transcript so the UI can credit them.
    result["best_line_by"] = find_speaker(result["best_line"], history, names)
    if human_side:
        result["your_strengths"] = clean_line(data.get("your_strengths")) or "You stayed on your side and kept your arguments on topic."
        result["improve"] = clean_line(data.get("improve")) or "Open by rebutting your opponent directly, and support each point with a real example or statistic."
    return jsonify(result)


# ---------------------------------------------------------------------------
# JSON helpers: ask for structured output, then parse it DEFENSIVELY
# ---------------------------------------------------------------------------

def ask_for_json(messages, required, temperature=0.2, max_tokens=1500, partial_ok=False):
    """Call the judge in JSON mode and parse the reply. Retries once on bad JSON.

    partial_ok: if keys are still missing after the retry, return what we got
    (the caller fills the gaps with safe defaults) instead of failing.
    """
    last_error = None
    partial = None
    for _attempt in range(2):  # first try + one retry
        try:
            raw = chat(messages, json_mode=True, temperature=temperature, max_tokens=max_tokens)
        except GroqError as err:
            if not err.json_failed:
                raise
            last_error = err  # Groq rejected invalid JSON -> retry
            continue
        data = parse_json_reply(raw)
        if data is not None and all(k in data for k in required):
            return data
        if data is not None:
            partial = data
        last_error = GroqError("Judge Sage replied in an unexpected format.", json_failed=True)
        print(f"[judge] reply missing keys or not JSON. Raw reply: {raw[:200]!r}")
    if partial_ok and partial is not None:
        return partial
    raise GroqError("Judge Sage's reply could not be read twice in a row. Please try again.",
                    status=502, json_failed=True) from last_error


def parse_json_reply(raw):
    """Turn the model's reply into a dict, even if it is a bit messy.

    1. strip ```json code fences   2. keep only the {...} part   3. json.loads
    """
    if not raw:
        return None
    text = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def clamp_score(value):
    """Force any score into a whole number from 1 to 10 (bad values -> 5)."""
    try:
        number = round(float(value))
    except (TypeError, ValueError):
        return 5
    return max(1, min(10, number))


# ---------------------------------------------------------------------------
# Text clean-up helpers
# ---------------------------------------------------------------------------

def clean_speech(raw, speaker_name, max_words=110):
    """Make sure a speech is plain text: remove markdown, labels and extra length."""
    text = raw or ""
    text = re.sub(r"[*_#`>]+", "", text)                       # markdown symbols
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)     # bullet points
    text = re.sub(rf"^\s*{speaker_name}\s*(\([^)]*\))?\s*:\s*", "", text, flags=re.I)  # "Nova:" label
    text = " ".join(text.split())                                # collapse whitespace/newlines
    text = text.strip('"“” ')

    words = text.split()
    if len(words) > max_words:
        # Too long: cut at the last full sentence inside the limit.
        cut = " ".join(words[:max_words])
        last_stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[: last_stop + 1] if last_stop > len(cut) // 2 else cut.rstrip(",;:") + "."
    return text


def clean_line(value):
    """Tidy a short string from the judge (or return '' if it isn't a string)."""
    if not isinstance(value, str):
        return ""
    return " ".join(re.sub(r"[*_#`]+", "", value).split())


def pick_best_line(history, winner):
    """Fallback best line: the longest sentence spoken by the winning side."""
    speeches = [h["text"] for h in history if winner == "draw" or h["side"] == winner]
    sentences = [s.strip() for t in speeches for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    return max(sentences, key=len) if sentences else ""


def find_speaker(line, history, names):
    """Return the name of whoever said `line` (or None if we can't find it)."""
    probe = line.lower().strip(" .!?")[:60]
    if not probe:
        return None
    for h in history:
        if probe in h["text"].lower():
            return names[h["side"]]
    return None


def default_reason(winner, names, totals):
    if winner == "draw":
        return f"Both sides finished level on {totals['for']} points. It was a genuinely even contest."
    loser = "against" if winner == "for" else "for"
    return (f"{names[winner]} won {totals[winner]} to {totals[loser]} on the judge's scorecards. "
            f"Their arguments were more consistently logical, supported and responsive.")


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def get_body():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError("Request body must be JSON.")
    return body


def validate_topic(topic):
    if not isinstance(topic, str):
        raise ApiError("Please enter a debate topic.")
    topic = " ".join(topic.split())
    if not TOPIC_MIN <= len(topic) <= TOPIC_MAX:
        raise ApiError(f"The topic must be between {TOPIC_MIN} and {TOPIC_MAX} characters.")
    return topic


def validate_side(side):
    if side not in ("for", "against"):
        raise ApiError('side must be "for" or "against".')
    return side


def validate_int(value, field, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ApiError(f"{field} must be a whole number.")
    value = int(value)
    if not low <= value <= high:
        raise ApiError(f"{field} must be between {low} and {high}.")
    return value


def validate_text(text, field):
    if not isinstance(text, str) or not text.strip():
        raise ApiError(f"{field} must not be empty.")
    return " ".join(text.split())[:SPEECH_MAX_CHARS]


def validate_history(history):
    """History = list of {side, round, text}. Keep only what we need, safely."""
    if history is None:
        return []
    if not isinstance(history, list) or len(history) > ROUNDS_MAX * 2:
        raise ApiError("history must be a list of at most 8 speeches.")
    clean = []
    for item in history:
        if not isinstance(item, dict):
            raise ApiError("Each history item must be an object.")
        clean.append({
            "side": validate_side(item.get("side")),
            "round": validate_int(item.get("round"), "history.round", 1, ROUNDS_MAX),
            "text": validate_text(item.get("text"), "history.text"),
        })
    return clean


def get_names(body):
    """Speaker names. In "You vs AI" mode the human takes one side's name."""
    names = {side: info["name"] for side, info in prompts.DEBATERS.items()}
    human_side = None
    if body.get("mode") == "human":
        human_side = validate_side(body.get("human_side"))
        names[human_side] = "You"
    return names, human_side


# ---------------------------------------------------------------------------
# Start the server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = load_settings()
    port = int(os.getenv("PORT", "5000"))
    line = "=" * 58
    print(f"\n{line}")
    print("  AI Debate Arena is live!")
    print(f"  Open this in your browser:  http://127.0.0.1:{port}")
    print(f"  Model: {settings['model']}  (fallback: {settings['fallback_model'] or 'none'})")
    if settings["api_key"]:
        print("  Groq API key: found in .env")
    else:
        print("  WARNING: no Groq API key yet. Add GROQ_API_KEY to your .env file")
        print("           (get a free key at https://console.groq.com/keys)")
    print("  Press CTRL+C to stop the server.")
    print(f"{line}\n")
    # threaded=True: the page prefetches, so e.g. a judge request and the next
    # speech request can run at the same time.
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
