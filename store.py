# pip install firecrawl-py
from firecrawl import Firecrawl

app = Firecrawl(api_key="")

# Scrape a website:
app.scrape('firecrawl.dev')