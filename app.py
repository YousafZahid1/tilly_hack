import asyncio
import json
import os
import re
from typing import Optional, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_api_client import LlamaAPIClient
from firecrawl import Firecrawl

load_dotenv()

firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
llama = LlamaAPIClient(api_key=os.getenv("LLAMA_API_KEY"),
                       base_url="https://api.llama.com/v1/")

LLAMA_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"
# Kept as aliases so existing call sites (model=...) keep working unchanged.
gemini = gemini_mid = gemini_lite = LLAMA_MODEL

from datetime import date
TODAY = date.today().isoformat()  # e.g. "2026-06-13" — models have no live clock

app = FastAPI()

MODE_DESC = {
    "read":    "Extracting & displaying data from the page",
    "agent":   "Firecrawl AI agent browsing autonomously",
    "browser": "Real browser session with live control",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def sse(type: str, **kw) -> str:
    return f"data: {json.dumps({'type': type, **kw})}\n\n"


def gemini_call(prompt: str, retries: int = 4, model=None,
                max_tokens: int = 4096) -> str:
    import time
    m = model or LLAMA_MODEL
    for attempt in range(retries):
        try:
            resp = llama.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
            return resp.completion_message.content.text
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                raise
    raise RuntimeError("LLM retries exhausted")


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[\w]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_markdown(scraped) -> str:
    if isinstance(scraped, dict):
        return (scraped.get("markdown") or scraped.get("content") or str(scraped))
    return getattr(scraped, "markdown", None) or str(scraped)


# ── intent classifier ──────────────────────────────────────────────────────────

def _classify_heuristic(url: str, task: str) -> str | None:
    """Fast keyword pre-classifier; returns None if ambiguous (falls through to Gemini)."""
    t = task.lower()
    browser_kws = {
        "book", "buy", "purchase", "order", "sign up", "signup", "register",
        "fill", "submit", "log in", "login", "apply", "reserve", "checkout",
        "find flights", "find hotels", "schedule",
    }
    read_kws = {
        "show", "list", "display", "extract", "get", "find", "top ", "summary",
        "summarize", "what are", "latest", "scrape",
    }
    if any(kw in t for kw in browser_kws):
        return "browser"
    if url and any(kw in t for kw in read_kws):
        return "read"
    return None


def _classify_sync(url: str, task: str) -> str:
    fast = _classify_heuristic(url, task)
    if fast:
        return fast

    prompt = f"""Classify this web task into ONE mode:

"read"    – user wants to VIEW / FIND / FILTER information already on a specific page
"agent"   – user wants to RESEARCH, COMPARE, or GATHER info across the web (no specific form to fill)
"browser" – user wants to BOOK, BUY, SIGN UP, FILL A FORM, or SUBMIT data on a real site

URL: {url or 'not given'}
Task: {task}

Reply with ONLY one word: read  agent  or  browser"""
    r_text = gemini_call(prompt, model=gemini_lite)
    m = r_text.strip().lower().split()[0]
    return m if m in ("read", "agent", "browser") else "agent"


# ── READ mode ──────────────────────────────────────────────────────────────────

def _firecrawl_extract(url: str, task: str):
    """Ask Firecrawl itself to extract structured data (uses Firecrawl credits,
    not the Gemini quota). Returns parsed JSON (dict/list) or None."""
    doc = firecrawl.scrape(url, formats=[{
        "type": "json",
        "prompt": f"Extract all data relevant to this goal: {task}. "
                  f"Return well-structured JSON with the key items as a list.",
    }])
    return getattr(doc, "json", None) or getattr(doc, "data", None)


def _template_ui(task: str, data) -> str:
    """Deterministic HTML renderer for structured JSON — no Gemini needed.
    Renders the main list of records as filterable cards."""
    # Find the primary list of records inside the JSON
    records, fields = [], []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                records = v
                break
        if not records:  # dict of scalars → single record
            records = [data]
    if records and isinstance(records[0], dict):
        fields = list(records[0].keys())

    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    cards = []
    for rec in records:
        if not isinstance(rec, dict):
            rec = {"value": rec}
        rows = []
        title = None
        for k in fields or rec.keys():
            val = rec.get(k, "")
            if title is None and isinstance(val, str) and val.strip():
                title = val
                continue
            disp = (f'<a href="{esc(val)}" target="_blank">{esc(val)}</a>'
                    if isinstance(val, str) and val.startswith("http")
                    else esc(val))
            rows.append(f'<div class="row"><span class="k">{esc(k)}</span>'
                        f'<span class="v">{disp}</span></div>')
        cards.append(f'<div class="card"><div class="title">{esc(title or "")}</div>'
                     f'{"".join(rows)}</div>')

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background:#fff; color:#1a1a1a; margin:0; padding:24px; max-width:860px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 16px; }}
#q {{ width:100%; padding:11px 14px; font-size:1rem; border:1px solid #ddd;
  border-radius:8px; margin-bottom:20px; outline:none; }}
#q:focus {{ border-color:#3b82f6; box-shadow:0 0 0 3px rgba(59,130,246,.15); }}
.card {{ border:1px solid #eee; border-radius:10px; padding:16px 18px; margin-bottom:12px;
  background:#fafafa; }}
.card:hover {{ box-shadow:0 4px 14px rgba(0,0,0,.06); }}
.title {{ font-weight:600; font-size:1.1rem; margin-bottom:8px; }}
.row {{ display:flex; gap:10px; font-size:.92rem; padding:2px 0; }}
.k {{ color:#888; min-width:120px; text-transform:capitalize; }}
.v {{ color:#222; }}
a {{ color:#2563eb; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
</style></head><body>
<h1>{esc(task)}</h1>
<input id="q" placeholder="Filter…" onkeyup="f()">
<div id="list">{"".join(cards) or "<p>No structured data found.</p>"}</div>
<script>
function f(){{var t=document.getElementById('q').value.toLowerCase();
document.querySelectorAll('.card').forEach(function(c){{
c.style.display=c.textContent.toLowerCase().includes(t)?'':'none';}});}}
</script></body></html>"""


async def _read_mode(url: str, task: str) -> AsyncIterator[str]:
    if not url:
        yield sse("error", msg="READ mode needs a URL.")
        return

    # Let Firecrawl extract structured data directly (no Gemini quota used).
    yield sse("step", msg=f"Extracting structured data from {url} with Firecrawl...")
    try:
        data = await asyncio.to_thread(_firecrawl_extract, url, task)
    except Exception as e:
        yield sse("error", msg=f"Firecrawl extraction failed: {e}")
        return

    if not data:
        yield sse("error", msg="No structured data returned from that URL.")
        return

    # Try the LLM for a custom AI-designed UI; fall back to a template on failure.
    yield sse("step", msg="Designing your interface with AI...")
    prompt = f"""You are a UI generator. The user's goal: {task}

Structured data (already extracted from {url}):
{json.dumps(data, indent=2)[:30000]}

Generate ONE self-contained HTML file:
- Shows ONLY data relevant to the goal, populated with the REAL data above
- Best layout: table, card grid, or result list
- All CSS in a <style> block — zero external deps
- Clean white background, system sans-serif, subtle borders
- Add a live JS filter/search input at top if helpful
- First line must be: <!DOCTYPE html>

Return ONLY raw HTML. No markdown fences."""
    try:
        r = await asyncio.to_thread(gemini_call, prompt, retries=2, model=gemini_mid)
        yield sse("html", html=strip_fences(r))
    except Exception as e:
        yield sse("step", msg=f"AI unavailable ({str(e)[:40]}…) — using built-in template.")
        yield sse("html", html=_template_ui(task, data))


# ── AGENT mode ─────────────────────────────────────────────────────────────────

async def _agent_mode(url: str, task: str) -> AsyncIterator[str]:
    yield sse("step", msg="Launching Firecrawl spark-1-pro agent...")

    try:
        started = await asyncio.to_thread(
            lambda: firecrawl.start_agent(
                urls=[url] if url else None,
                prompt=task,
                model="spark-1-pro",
            )
        )
    except Exception as e:
        yield sse("error", msg=f"Agent launch failed: {e}")
        return

    job_id = getattr(started, "id", None)
    if not job_id:
        yield sse("error", msg="Agent returned no job ID.")
        return

    yield sse("step", msg=f"Agent running (job {job_id[:8]}...) — may take 60–120s")

    elapsed = 0
    polls   = 0
    while True:
        await asyncio.sleep(10)
        elapsed += 10
        polls   += 1
        try:
            status = await asyncio.to_thread(firecrawl.get_agent_status, job_id)
        except Exception as e:
            yield sse("error", msg=f"Status poll failed: {e}")
            return

        state = getattr(status, "status", "processing")
        if polls % 3 == 0:  # log once every 30s instead of every 4s
            yield sse("step", msg=f"Still working... ({elapsed}s elapsed)")

        if state in ("completed", "failed", "cancelled"):
            break

    if state != "completed":
        yield sse("error", msg=f"Agent ended with status: {state}")
        return

    data = getattr(status, "data", None)
    if not data:
        yield sse("error", msg="Agent completed but returned no data.")
        return

    result_str = json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)
    yield sse("step", msg="Rendering agent results with AI...")

    prompt = f"""The user wanted: {task}

A Firecrawl AI agent completed this task and returned:
{result_str[:30000]}

Generate a clean self-contained HTML page presenting these results:
- Lead with a 1-sentence summary of what was accomplished
- Show all data in the best layout for the content (table, cards, list)
- All CSS inline — no external deps
- Clean white bg, system sans-serif, good typography
- First line: <!DOCTYPE html>

Return ONLY raw HTML."""

    try:
        r = await asyncio.to_thread(gemini_call, prompt, model=gemini_mid)
        yield sse("html", html=strip_fences(r))
    except Exception as e:
        yield sse("error", msg=f"Render failed: {e}")


# ── BROWSER mode — Phase 1: generate a form mirroring the real site ─────────────

async def _browser_mode(url: str, task: str) -> AsyncIterator[str]:
    """Phase 1: scrape the target site and generate a minimal form the user
    fills in. No browser session is created yet (saves rate limit / credits).
    The actual real-site automation happens in Phase 2 via /run-action."""

    if not url:
        yield sse("step", msg="No URL given — AI will infer the right site.")
        md = ""
    else:
        yield sse("step", msg=f"Inspecting {url} to find the inputs it needs...")
        try:
            scraped = await asyncio.to_thread(firecrawl.scrape, url)
            md = extract_markdown(scraped)[:30000]
        except Exception as e:
            yield sse("step", msg=f"Could not scrape ({e}); generating form from task alone.")
            md = ""

    yield sse("step", msg="Building a clean input form for your task...")

    form_prompt = f"""You are a UI generator for a "No-UI Browser". Today's date is {TODAY}.
The user's goal:
{task}

{('Here is the real page content so you know what inputs the site needs:' + chr(10) + '--- BEGIN ---' + chr(10) + md + chr(10) + '--- END ---') if md else 'No page content available — infer the fields a user would need to provide for this task.'}

Generate ONE self-contained HTML form that collects ONLY the inputs needed to
accomplish the goal on the real site (e.g. for a flight: from, to, date,
passengers; for Airbnb: city, dates, guests). Rules:
- Clean minimal form: labels + inputs, a single "Continue" submit button
- Pick correct input types (date, number, text, select with real options if known)
- For any date input, set min to {TODAY} and pre-fill a sensible future date
- Pre-fill values that are already stated in the user's goal
- DO NOT ask for passwords or payment — those are handled later in the live browser
- All CSS in a <style> block. White bg, system sans-serif. No external deps.
- First line must be: <!DOCTYPE html>

CRITICAL — include this exact submit script so the parent app receives the values:
<script>
document.querySelector('form').addEventListener('submit', function(e) {{
  e.preventDefault();
  const values = {{}};
  new FormData(e.target).forEach((v, k) => values[k] = v);
  window.parent.postMessage({{ type: 'form-submit', values }}, '*');
}});
</script>

Return ONLY raw HTML. No markdown fences. No commentary."""

    try:
        r = await asyncio.to_thread(gemini_call, form_prompt, model=gemini_mid)
        yield sse("form", html=strip_fences(r), url=url, task=task)
    except Exception as e:
        yield sse("error", msg=f"Form generation failed: {e}")


# ── BROWSER mode — Phase 2: drive the real site with the user's data ────────────

def _build_automation_code(user_data: dict, task: str,
                           url: str, mode: str) -> str:
    """Synchronous Gemini call that returns async-Playwright code (verified to
    run in the sandbox's asyncio event loop)."""

    if mode == "self":
        handoff_rule = (
            "- The user wants to TAKE OVER. Navigate to the correct page and fill "
            "only the obvious search fields, then STOP before any login/booking/"
            "payment. Print 'STEP: Ready for you — take over the live browser now.'"
        )
    else:
        handoff_rule = (
            "- The user wants the AGENT to proceed. Fill every field you can from "
            "user_data and advance through the flow. When you reach a step needing "
            "a password or payment, STOP and print "
            "'STEP: Paused — enter your password/payment in the live browser.' "
            "Do NOT invent secret credentials or card numbers."
        )

    plan_prompt = f"""You are an expert web-automation engineer. Write ASYNC Playwright
Python that drives a REAL website to help the user accomplish their goal.

Today's date is {TODAY}. Use it for any relative dates (e.g. "next week", "tomorrow").
Goal: {task}
Starting URL: {url or 'pick the best site and navigate there'}

These two variables ALREADY EXIST in the runtime — use them, never redefine them:
  cdp_url   = CDP websocket URL for the live browser (string)
  user_data = {json.dumps(user_data)}   ← the REAL values the user typed

HARD RULES (breaking any of these makes the script crash — follow exactly):
1. The code runs INSIDE a running asyncio loop. Use async/await everywhere.
   NEVER use sync_playwright, asyncio.run(), browser.close(), exit(), sys.exit(),
   quit(), or return at top level.
2. EVERY page interaction (goto/fill/click/press/wait_for_selector) MUST be wrapped
   in its own try/except so a single failure NEVER aborts the script. On failure,
   print("STEP: <what failed, in plain words>") and continue to the next step.
3. print("STEP: ...") before each meaningful action so the user sees progress.
4. The script must always reach its final line and print the JSON summary.

Start with EXACTLY this header (do not add cdp_url/user_data assignments):

from playwright.async_api import async_playwright
import json
p = await async_playwright().start()
browser = await p.chromium.connect_over_cdp(cdp_url)
context = browser.contexts[0] if browser.contexts else await browser.new_context()
page = context.pages[0] if context.pages else await context.new_page()

Here is the REQUIRED style for every step (copy this pattern exactly):

print("STEP: Open the site")
try:
    await page.goto("https://example.com", timeout=30000)
    await page.wait_for_timeout(1500)
except Exception as e:
    print(f"STEP: Could not open the site ({{e}})")

try:
    await page.click("button:has-text('Accept')", timeout=4000)
except Exception:
    pass

The LAST line must ALWAYS be exactly:
print(json.dumps({{"status":"done","summary":"<one sentence>","data":{{}}}}))

{handoff_rule}

Return ONLY raw Python. No markdown fences. No explanatory comments."""

    r_text = gemini_call(plan_prompt, max_tokens=4096)
    code = strip_fences(r_text)

    safe_lines = []
    for line in code.splitlines():
        # Drop any reassignment of the injected globals
        if re.match(r"^\s*(cdp_url|user_data)\s*=", line):
            continue
        indent = line[:len(line) - len(line.lstrip())]
        stripped = line.strip()
        # Neutralize calls that would abort the whole sandbox script
        if re.match(r"^(sys\.)?exit\s*\(|^quit\s*\(", stripped):
            safe_lines.append(f"{indent}pass")
            continue
        if "browser.close()" in stripped or "asyncio.run" in stripped:
            safe_lines.append(f"{indent}pass")
            continue
        safe_lines.append(line)
    code = "\n".join(safe_lines)

    # Guard against truncated/invalid code reaching the sandbox: it runs inside an
    # async wrapper, so validate by compiling it wrapped in `async def`.
    import ast
    try:
        ast.parse("async def __m():\n" + "\n".join("    " + l for l in code.splitlines()))
    except SyntaxError:
        # Fall back to a minimal, always-valid navigate-and-handoff script.
        target = url or "https://duckduckgo.com"
        code = (
            "from playwright.async_api import async_playwright\n"
            "import json\n"
            "p = await async_playwright().start()\n"
            "browser = await p.chromium.connect_over_cdp(cdp_url)\n"
            "context = browser.contexts[0] if browser.contexts else await browser.new_context()\n"
            "page = context.pages[0] if context.pages else await context.new_page()\n"
            'print("STEP: Opening the site for you")\n'
            "try:\n"
            f'    await page.goto({json.dumps(target)}, timeout=30000)\n'
            "except Exception as e:\n"
            '    print(f"STEP: Could not open the site ({e})")\n'
            'print("STEP: Ready for you — take over the live browser now.")\n'
            'print(json.dumps({"status":"done","summary":"Opened the site for manual takeover","data":{}}))\n'
        )
    return code


async def _run_action(url: str, task: str, user_data: dict, mode: str) -> AsyncIterator[str]:
    yield sse("step", msg="Creating live browser session...")
    try:
        session = await asyncio.to_thread(
            lambda: firecrawl.browser(ttl=600, stream_web_view=True)
        )
    except Exception as e:
        msg = str(e)
        if "Rate limit" in msg or "RateLimit" in type(e).__name__:
            yield sse("error", msg="Firecrawl browser rate limit (3/min). Wait ~30s and retry.")
        else:
            yield sse("error", msg=f"Browser session failed: {e}")
        return

    session_id = getattr(session, "id", None)
    cdp_url    = getattr(session, "cdp_url", None)
    live_url   = (getattr(session, "interactive_live_view_url", None)
                  or getattr(session, "live_view_url", None))

    if live_url:
        yield sse("live_view", url=live_url,
                  msg="Live browser is open — you can click in it anytime")

    yield sse("step", msg="Planning the automation with your data...")
    try:
        code = await asyncio.to_thread(
            _build_automation_code, user_data, task, url, mode
        )
    except Exception as e:
        yield sse("error", msg=f"Failed to plan automation: {e}")
        await _kill_browser(session_id)
        return

    injected = f"cdp_url = {json.dumps(cdp_url or '')}\nuser_data = {json.dumps(user_data)}\n\n{code}"

    yield sse("step", msg="Driving the real website...")
    stdout = ""
    try:
        exec_resp = await asyncio.to_thread(
            lambda: firecrawl.browser_execute(
                session_id, injected, language="python", timeout=120
            )
        )
        stdout = (getattr(exec_resp, "stdout", None)
                  or getattr(exec_resp, "output", None)
                  or getattr(exec_resp, "result", None) or "")
        stderr = getattr(exec_resp, "stderr", None) or ""
        exit_code = getattr(exec_resp, "exit_code", None)

        for line in stdout.splitlines():
            s = line.strip()
            if s.startswith("STEP:"):
                yield sse("step", msg=s[5:].strip())

        if exit_code not in (0, None):
            yield sse("step", msg=f"Automation exited with code {exit_code}.")

        yield sse("step", msg="Rendering the outcome...")
        render_prompt = f"""The user wanted: {task}
They provided: {json.dumps(user_data)}

Browser automation output:
--- STDOUT ---
{stdout[:8000]}
--- STDERR ---
{stderr[:2000]}

Generate a self-contained HTML page that:
- States clearly what was accomplished on the real site
- Shows any results/data found (prices, confirmations, options)
- If it paused for the user (password/payment), say so and tell them to finish
  in the live browser tab
- Clean white bg, system sans-serif. First line: <!DOCTYPE html>

Return ONLY raw HTML."""
        r2 = await asyncio.to_thread(gemini_call, render_prompt, model=gemini_mid)
        yield sse("html", html=strip_fences(r2))

    except Exception as e:
        yield sse("error", msg=f"Execution error: {e}")

    # Keep the session alive for human takeover; it expires via its 600s TTL.
    if mode == "self" or "paus" in stdout.lower() or "take over" in stdout.lower():
        yield sse("step", msg="Live browser stays open for you to finish.")
    else:
        await _kill_browser(session_id)


async def _kill_browser(session_id: str):
    try:
        await asyncio.to_thread(lambda: firecrawl.delete_browser(session_id))
    except Exception:
        pass


# ── Main execute endpoint (SSE) ────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    url: Optional[str] = ""
    task: str


@app.post("/execute")
async def execute(req: ExecuteRequest):
    url  = (req.url or "").strip()
    task = req.task.strip()

    if not task:
        raise HTTPException(status_code=400, detail="task is required")

    async def stream():
        try:
            yield sse("step", msg="Analyzing your task...")
            mode = await asyncio.to_thread(_classify_sync, url, task)
            yield sse("mode", mode=mode, label=MODE_DESC.get(mode, mode))
            yield sse("step", msg=f"Mode selected: {mode.upper()} — {MODE_DESC[mode]}")

            if mode == "read":
                async for ev in _read_mode(url, task):
                    yield ev
            elif mode == "agent":
                async for ev in _agent_mode(url, task):
                    yield ev
            else:
                async for ev in _browser_mode(url, task):
                    yield ev

            yield sse("done", msg="Task complete.")

        except Exception as e:
            yield sse("error", msg=f"Unexpected error: {e}")

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Phase 2: run the real-site automation with the user's form data ─────────────

class RunActionRequest(BaseModel):
    url: Optional[str] = ""
    task: str
    user_data: dict = {}
    mode: str = "agent"  # "agent" | "self"


@app.post("/run-action")
async def run_action(req: RunActionRequest):
    url  = (req.url or "").strip()
    task = req.task.strip()
    mode = req.mode if req.mode in ("agent", "self") else "agent"

    if not task:
        raise HTTPException(status_code=400, detail="task is required")

    async def stream():
        try:
            yield sse("mode", mode="browser", label=MODE_DESC["browser"])
            async for ev in _run_action(url, task, req.user_data, mode):
                yield ev
            yield sse("done", msg="Action complete.")
        except Exception as e:
            yield sse("error", msg=f"Unexpected error: {e}")

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Legacy simple endpoint ────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    url: str
    task: str


@app.post("/generate-ui")
async def generate_ui(req: GenerateRequest):
    try:
        scraped = await asyncio.to_thread(firecrawl.scrape, req.url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Firecrawl error: {e}")

    md = extract_markdown(scraped)[:50000]
    if not md.strip():
        raise HTTPException(status_code=422, detail="No content returned.")

    prompt = f"""Goal: {req.task}

Content from {req.url}:
{md}

Generate one self-contained HTML with only data relevant to the goal.
Real data from above. <!DOCTYPE html> first. No external deps.
Return ONLY raw HTML."""

    r = await asyncio.to_thread(gemini_call, prompt)
    return {"html": strip_fences(r)}


# ── Static serving ────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
