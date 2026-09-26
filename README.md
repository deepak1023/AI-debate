# AI Debate Arena

Two AI debaters argue any topic you give them, live on a virtual stage:

- **Nova** argues **for** the motion
- **Rex** argues **against** it
- **Judge Sage**, a third AI, scores every round on logic, evidence and rebuttal, then declares the winner

In **You vs AI** mode you take one side yourself, the AI argues the other, and Judge Sage scores you and gives you feedback. It works as practice for group discussions.

Built with Python + Flask, plain HTML/CSS/JavaScript and the free [Groq API](https://console.groq.com) running OpenAI's open-weight `gpt-oss-120b` model (with `gpt-oss-20b` as a fallback). It can also be deployed to Netlify for free (see [Deploy to Netlify](#deploy-to-netlify)).

---

## Project structure

```
ai-debate-arena/
  app.py              Flask server: routes, validation, winner calculation
  groq_client.py      Calls the Groq API (retry on rate limits, fallback model)
  prompts.py          All AI prompts (Nova, Rex, Judge Sage) in one place
  requirements.txt    Python libraries: flask, requests, python-dotenv
  .env.example        Template for your secret settings
  public/             The website (shared by Flask AND Netlify)
    index.html
    css/style.css
    js/app.js
  netlify/            Netlify version of the backend (JavaScript)
    functions/        One serverless function per endpoint
    lib/              Shared code: prompts, Groq client, validation
  netlify.toml        Netlify settings
```

> The frontend lives in `public/` (instead of `templates/` + `static/`) so the **same** files work locally with Flask and online with Netlify.

---

## Run it locally (VS Code)

### 1. Install Python

- **Windows:** download Python 3.10 or newer from [python.org](https://www.python.org/downloads/). In the installer, **tick "Add python.exe to PATH"**.
- **Mac:** download it from [python.org](https://www.python.org/downloads/), or run `brew install python`.

Check it works in a terminal: `python --version` (on Mac: `python3 --version`).

### 2. Open the project in VS Code

`File > Open Folder…` and pick the `ai-debate-arena` folder. Open a terminal with `` Ctrl+` `` (Mac: `` Cmd+` ``).

### 3. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```
If PowerShell says "running scripts is disabled", run this once, then try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` at the start of the terminal line when it's active.

### 4. Install the libraries

```bash
pip install -r requirements.txt
```

### 5. Add your free Groq API key

1. Copy the example settings file:
   - Windows: `copy .env.example .env`
   - Mac: `cp .env.example .env`
2. Go to [console.groq.com/keys](https://console.groq.com/keys), sign in (free), click **Create API Key** and copy it.
3. Open `.env` in VS Code and paste your key:
   ```
   GROQ_API_KEY=gsk_your_real_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   GROQ_FALLBACK_MODEL=openai/gpt-oss-20b
   ```

`.env` is listed in `.gitignore`, so your key is never uploaded to GitHub. The key stays on the server and is never sent to the browser.

### 6. Start the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. To stop the server, press `Ctrl+C` in the terminal.

---

## How it works (the GenAI concepts)

| Concept | Where it happens in this project |
|---|---|
| **Multi-agent system** | Three AI "agents" (Nova, Rex, Judge Sage) each have a separate job and work together. `public/js/app.js` is the *orchestrator*: it decides who speaks next and passes results between the agents. |
| **Role prompting** | All three agents are the **same** gpt-oss model. They behave differently only because each gets its own *system prompt* describing its character, side and rules (`prompts.py`). |
| **Context / history management** | LLMs have no memory between API calls. So every time a debater speaks, we send the **full debate transcript** plus the opponent's latest point. That's how Nova and Rex actually answer each other. |
| **Structured JSON output** | The judge must reply in an exact JSON shape. We use Groq's `response_format: {"type": "json_object"}` **and** still parse defensively: strip code fences, extract the `{...}`, clamp scores to 1–10 and retry once if parsing fails. |
| **Reasoning model** | gpt-oss "thinks" before it answers, and that hidden reasoning also counts toward `max_tokens`. Every request sends `reasoning_effort: "low"` and a generous `max_tokens` (1500 for speeches and round scores, 2000 for the verdict), so the final answer is never cut off. We read only `message.content` and ignore the separate `reasoning` field. If the answer is empty, we retry once before showing an error. |
| **LLM-as-a-judge** | Judge Sage evaluates the other AIs' output against a rubric (logic, evidence, rebuttal). The **winner is calculated in Python** from the score totals; the AI only explains the result. We never trust an LLM to do maths. |

Other details worth mentioning to the judges:
- **Temperature:** debaters use 0.85 (creative, lively), the judge uses 0.2 (consistent, fair).
- **Prompt-injection guard:** prompts tell the model to treat the transcript as content, not instructions. The frontend inserts AI text with `textContent` (never `innerHTML`), so replies can't inject HTML.
- **Reliability:** input validation, automatic retry on rate limits, an automatic fallback to a smaller model, and a "Try again" button in the UI if a step fails.
- **Prefetching:** each API call starts the moment its inputs exist, not when the screen gets there. The next speech is fetched while the current one is still being typed out, and Judge Sage scores a round while its last speech is still on its way to the screen. Results are always shown in order. Press **Skip** on a card to see the whole speech at once.
- **Finding delays:** the browser console (F12) logs how long every API call took, and when the screen had to wait for one. The terminal logs every rate-limit (429) retry, with Groq's reason and how many tokens and requests you have left this minute.

### API usage per debate

| Mode | Calls | Example (3 rounds) |
|---|---|---|
| AI vs AI | rounds × 3 + 1 (2 speeches + 1 judge per round, + 1 verdict) | 10 calls |
| You vs AI | rounds × 2 + 1 (your speech costs nothing) | 7 calls |

Groq's free tier comfortably covers this for demos.

---

## API endpoints

| Method | URL | Body | Returns |
|---|---|---|---|
| GET | `/api/status` | – | `{model, key_configured}` |
| POST | `/api/speech` | `{topic, side, round, total_rounds, history}` | `{text}` |
| POST | `/api/judge-round` | `{topic, round, for_text, against_text}` | `{for:{logic,evidence,rebuttal,total}, against:{…}, comment}` |
| POST | `/api/verdict` | `{topic, history, totals, mode}` | `{winner, totals, reason, best_line, tip}` (+ `your_strengths`, `improve` in You vs AI) |

---

## Deploy to Netlify

The `netlify/functions/` folder has a JavaScript copy of the backend: one function each for `status`, `speech`, `judge-round` and `verdict`, with the same JSON in and out. `netlify.toml` redirects `/api/*` to those functions, so the frontend works unchanged. The functions use the built-in `fetch` in Node 18+, so there's nothing to `npm install`.

### Steps

1. **Push the project to GitHub.** In the `ai-debate-arena` folder:
   ```bash
   git init
   git add .
   git commit -m "AI Debate Arena"
   ```
   Create an empty repository on [github.com/new](https://github.com/new), then run the two commands GitHub shows you (`git remote add origin …` and `git push -u origin main`).
   `.env` is in `.gitignore`, so your key is **not** uploaded. Check this on GitHub before you continue.
2. **Import the repo in Netlify.** Log in at [app.netlify.com](https://app.netlify.com) → **Add new site → Import an existing project → GitHub** → pick your repository. Netlify reads `netlify.toml` automatically. Leave the build command empty.
3. **Add your key.** Go to **Site configuration → Environment variables → Add a variable**:
   - `GROQ_API_KEY` = your key
   - `GROQ_MODEL` = `openai/gpt-oss-120b` (optional)
   - `GROQ_FALLBACK_MODEL` = `openai/gpt-oss-20b` (optional)
4. **Deploy.** Go to **Deploys → Trigger deploy → Deploy site**. You need to redeploy after changing environment variables.
5. Open your `https://your-site.netlify.app` link. The model badge in the top bar should show a green dot.

### Test the Netlify version locally with `netlify dev`

Requires [Node.js 18+](https://nodejs.org).
```bash
npm install -g netlify-cli
netlify dev
```
This runs the site plus the functions at **http://localhost:8888**, reading your key from `.env`.

**Which one should I use locally?** For your demo, `python app.py` is the simplest and is fully supported. Use `netlify dev` only when you want to test the Netlify functions before deploying. Both use the same frontend and the same prompts.

### How the Netlify version protects your free quota

- Rejects empty requests, bodies over 32 KB, topics outside 3–200 characters, rounds outside 2–4 and histories longer than 8 speeches.
- Only accepts browser requests coming from your own site (an origin check).
- Best-effort rate limit of 60 requests per 10 minutes per visitor IP. Change it with a `RATE_LIMIT_MAX` environment variable.
- Each function makes one Groq request, with a 9-second time budget so it finishes inside Netlify's 10-second limit. Rate-limit retries only happen while time remains.

> If you change a prompt, change it in both `prompts.py` and `netlify/lib/prompts.js`.

---

## Troubleshooting

**"API key missing: add GROQ_API_KEY to your .env file"**
- Make sure the file is called exactly `.env` (not `.env.txt` or `.env.example`) and sits next to `app.py`.
- The line must be `GROQ_API_KEY=gsk_...` with no quotes or spaces around `=`.
- Refresh the page. The app re-reads `.env` on every request, so you don't need to restart.
- On Netlify: add the variable in Site configuration → Environment variables, then **redeploy**.

**"Groq rate limit reached"**
- The free tier limits requests and tokens per minute. The app waits and retries automatically (up to 3 times), then switches to the smaller `openai/gpt-oss-20b` model if needed. Each retry is logged in the terminal as `[groq] RATE LIMIT (429) ...`, along with the limit you hit.
- If it still fails, wait about 30 seconds and press **Try again**. Your debate isn't lost.
- For a smoother live demo, use 2 or 3 rounds, or set `GROQ_MODEL=openai/gpt-oss-20b` in `.env`. It's smaller and faster.

**"Groq error (404): The model … does not exist" / "has been decommissioned"**
- Groq retires old models from time to time. Open `.env` and make sure `GROQ_MODEL` and `GROQ_FALLBACK_MODEL` use current names (see `.env.example`), or delete those two lines to use the built-in defaults. On Netlify, update the environment variables and redeploy.

**"The AI sent back an empty answer"**
- The model used all its tokens on hidden reasoning and wrote no final answer, even after one automatic retry. Press **Try again**.

**"Address already in use" / port 5000 is busy**
- Another program is using port 5000. On Mac, AirPlay Receiver often uses it: turn it off in System Settings → General → AirDrop & Handoff.
- Or pick another port by adding `PORT=5050` to `.env`, then open http://127.0.0.1:5050.
- Or close the old server: find the other terminal running `python app.py` and press `Ctrl+C`.

**`python` is not recognised (Windows)**
Reinstall Python and tick "Add python.exe to PATH", or use `py app.py` instead.

**The page loads but says "Can't reach the server"**
The Flask server has stopped. Run `python app.py` again and refresh.

**Fonts look plain**
The fonts load from Google Fonts. Without internet, the app falls back to system fonts. It still works, but you need internet for Groq anyway.
