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
import google.generativeai as genai
from firecrawl import Firecrawl

load_dotenv()

firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

MODE_DESC = {
    "read":    "Extracting & displaying data from the page",
    "agent":   "Firecrawl AI agent browsing autonomously",
    "browser": "Real browser session with live control",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def sse(type: str, **kw) -> str:
    return f"data: {json.dumps({'type': type, **kw})}\n\n"


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

def _classify_sync(url: str, task: str) -> str:
    prompt = f"""Classify this web task into ONE mode:

"read"    – user wants to VIEW / FIND / FILTER information already on a specific page
"agent"   – user wants to RESEARCH, COMPARE, or GATHER info across the web (no specific form to fill)
"browser" – user wants to BOOK, BUY, SIGN UP, FILL A FORM, or SUBMIT data on a real site

URL: {url or 'not given'}
Task: {task}

Reply with ONLY one word: read  agent  or  browser"""
    r = gemini.generate_content(prompt)
    m = r.text.strip().lower().split()[0]
    return m if m in ("read", "agent", "browser") else "agent"


# ── READ mode ──────────────────────────────────────────────────────────────────

async def _read_mode(url: str, task: str) -> AsyncIterator[str]:
    if not url:
        yield sse("error", msg="READ mode needs a URL.")
        return

    yield sse("step", msg=f"Scraping {url} with Firecrawl...")
    try:
        scraped = await asyncio.to_thread(firecrawl.scrape, url)
    except Exception as e:
        yield sse("error", msg=f"Scrape failed: {e}")
        return

    md = extract_markdown(scraped)[:50000]
    if not md.strip():
        yield sse("error", msg="No content returned from that URL.")
        return

    yield sse("step", msg="Building your custom interface with Gemini...")

    prompt = f"""You are a UI generator. The user's goal: {task}

Raw content from {url}:
--- BEGIN ---
{md}
--- END ---

Generate ONE self-contained HTML file:
- Shows ONLY data relevant to the user's goal
- Pick the best layout: table, card grid, or result list
- All CSS in a <style> block — zero external deps
- Populate with REAL data from the content above (no placeholders)
- No nav bars, ads, footers, login prompts, or unrelated chrome
- Clean white background, system sans-serif font, subtle borders
- Add a live JS filter/search input at top if that helps the task
- First line must be: <!DOCTYPE html>

Return ONLY raw HTML. No markdown fences. No commentary."""

    try:
        r = await asyncio.to_thread(gemini.generate_content, prompt)
        yield sse("html", html=strip_fences(r.text))
    except Exception as e:
        yield sse("error", msg=f"Gemini error: {e}")


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

    yield sse("step", msg=f"Agent running (job {job_id[:8]}...)  — this may take 30–90s")

    elapsed = 0
    while True:
        await asyncio.sleep(4)
        elapsed += 4
        try:
            status = await asyncio.to_thread(firecrawl.get_agent_status, job_id)
        except Exception as e:
            yield sse("error", msg=f"Status poll failed: {e}")
            return

        state = getattr(status, "status", "processing")
        yield sse("step", msg=f"Agent working... ({elapsed}s elapsed, status: {state})")

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
    yield sse("step", msg="Rendering agent results with Gemini...")

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
        r = await asyncio.to_thread(gemini.generate_content, prompt)
        yield sse("html", html=strip_fences(r.text))
    except Exception as e:
        yield sse("error", msg=f"Render failed: {e}")


# ── BROWSER mode ───────────────────────────────────────────────────────────────

async def _browser_mode(url: str, task: str) -> AsyncIterator[str]:
    yield sse("step", msg="Creating live browser session...")

    try:
        session = await asyncio.to_thread(
            lambda: firecrawl.browser(ttl=600, stream_web_view=True)
        )
    except Exception as e:
        yield sse("error", msg=f"Browser session failed: {e}")
        return

    session_id = getattr(session, "id", None)
    cdp_url    = getattr(session, "cdp_url", None)
    live_url   = (getattr(session, "interactive_live_view_url", None)
                  or getattr(session, "live_view_url", None))

    if live_url:
        yield sse("live_view", url=live_url,
                  msg="Live browser is open — watch below, you can interact anytime")

    yield sse("step", msg="Generating step-by-step automation plan with Gemini...")

    # Gemini generates Playwright Python for the task
    plan_prompt = f"""Write Playwright Python code to complete this browser task.

Task: {task}
Starting URL: {url or 'navigate to the appropriate site'}

The code runs inside a sandbox that already has:
- Python 3.11 + playwright installed
- A variable `cdp_url` (string) pointing to the live Chromium browser via CDP

Your code MUST follow this exact structure:

from playwright.sync_api import sync_playwright
import json, time

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp(cdp_url)
    context = browser.contexts[0] if browser.contexts else browser.new_context()
    page = context.pages[0] if context.pages else context.new_page()

    # ── YOUR STEPS HERE ──
    # Use print("STEP: <description>") to report each major step
    # At the very end print a JSON summary:
    # print(json.dumps({{"status": "done", "summary": "what happened", "data": {{}}}}))

Rules:
- Use page.goto(), page.fill(), page.click(), page.wait_for_selector()
- Accept cookie banners: try: page.click("button[aria-label*='Accept']", timeout=3000) except: pass
- Add page.wait_for_timeout(1500) between major interactions
- NEVER call browser.close() or playwright.stop()
- Handle errors with try/except and print an error JSON if something fails
- If the task needs user input data not specified (e.g. passenger name, credit card), print a STEP saying what's missing and stop gracefully

Return ONLY raw Python code. No markdown fences."""

    try:
        r = await asyncio.to_thread(gemini.generate_content, plan_prompt)
        code = strip_fences(r.text)
    except Exception as e:
        yield sse("error", msg=f"Failed to generate plan: {e}")
        await _kill_browser(session_id)
        return

    # Inject the cdp_url as the first line so generated code can use it
    injected_code = f"cdp_url = {json.dumps(cdp_url or '')}\n\n{code}"

    yield sse("step", msg="Executing automation in the live browser...")

    try:
        exec_resp = await asyncio.to_thread(
            lambda: firecrawl.browser_execute(
                session_id, injected_code, language="python", timeout=120
            )
        )
        stdout = (getattr(exec_resp, "stdout", None)
                  or getattr(exec_resp, "output", None)
                  or getattr(exec_resp, "result", None)
                  or "")
        stderr = getattr(exec_resp, "stderr", None) or ""
        exit_code = getattr(exec_resp, "exit_code", None)

        # Stream STEP progress lines from stdout
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("STEP:"):
                yield sse("step", msg=stripped[5:].strip())

        combined = (stdout + "\n" + stderr).strip()
        yield sse("step", msg=f"Automation finished (exit {exit_code}). Rendering outcome...")

        render_prompt = f"""The user wanted: {task}

Browser automation produced this output:
--- STDOUT ---
{stdout[:8000]}
--- STDERR ---
{stderr[:2000]}

Generate a self-contained HTML page that:
- Clearly states what was accomplished (or what's missing / needs user input)
- Shows any data/results found
- If something is missing (e.g. payment details), shows a friendly prompt explaining what the user needs to provide next
- Clean white bg, system sans-serif
- First line: <!DOCTYPE html>

Return ONLY raw HTML."""

        r2 = await asyncio.to_thread(gemini.generate_content, render_prompt)
        yield sse("html", html=strip_fences(r2.text))

    except Exception as e:
        yield sse("error", msg=f"Execution error: {e}")
    finally:
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

    r = await asyncio.to_thread(gemini.generate_content, prompt)
    return {"html": strip_fences(r.text)}


# ── Static serving ────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
