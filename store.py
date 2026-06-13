# pip install firecrawl-py
from firecrawl import Firecrawl

app = Firecrawl(api_key="fc-73de9961866d4d6db843e857eb02b43d")

# Scrape a website:
app.scrape('firecrawl.dev')