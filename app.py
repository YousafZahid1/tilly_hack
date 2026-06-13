import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
from firecrawl import Firecrawl

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()


class GenerateRequest(BaseModel):
    url: str
    task: str


@app.post("/generate-ui")
async def generate_ui(req: GenerateRequest):
    # Scrape the URL with Firecrawl
    try:
        scraped = firecrawl.scrape(req.url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Firecrawl error: {e}")

    # Extract markdown content, cap at 50k chars to stay within Gemini limits
    markdown = ""
    if isinstance(scraped, dict):
        markdown = scraped.get("markdown", "") or scraped.get("content", "") or str(scraped)
    else:
        markdown = getattr(scraped, "markdown", "") or str(scraped)

    markdown = markdown[:50000]

    if not markdown.strip():
        raise HTTPException(status_code=422, detail="Firecrawl returned no content for that URL.")

    prompt = f"""You are a UI generator. The user's goal is: {req.task}

Here is the raw content scraped from: {req.url}
--- BEGIN PAGE CONTENT ---
{markdown}
--- END PAGE CONTENT ---

Generate a SINGLE self-contained HTML file that:
- Shows ONLY the data directly relevant to the user's goal
- Uses the best layout for the task: a filtered table, card grid, or result list
- Has all CSS in a <style> block — NO external stylesheets, NO CDN links
- Populates REAL data extracted from the page content above (not placeholder text)
- Has ZERO navigation bars, login forms, ads, footers, or unrelated chrome
- Uses a clean white background, Inter or system sans-serif font, subtle gray borders
- Is immediately actionable — the user can accomplish their goal from this page alone
- If filtering/sorting is useful, add a small inline JS filter input at the top

Return ONLY the raw HTML. No markdown fences. No explanation. Start with <!DOCTYPE html>."""

    try:
        response = model.generate_content(prompt)
        html = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    # Strip any accidental markdown fences Gemini adds
    html = re.sub(r"^```html\s*", "", html)
    html = re.sub(r"^```\s*", "", html)
    html = re.sub(r"\s*```$", "", html)

    return {"html": html}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
