import os
import sys
import logging
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import anthropic
import chromadb
import voyageai
import json

load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger(__name__)

mcp = FastMCP("Anon0107-tools")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
CHROMA_API_KEY = os.getenv('CHROMA_API_KEY')
CHROMA_DATABASE = os.getenv('CHROMA_DATABASE')
CHROMA_TENANT = os.getenv('CHROMA_TENANT')
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')

@mcp.tool()
async def search_news(query: str, country: str = "my") -> str:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": 5,
        "sortBy": "publishedAt",
    }
    headlines_url = "https://newsapi.org/v2/top-headlines"
    headlines_params = {
        "q": query,
        "country": country,
        "apiKey": NEWS_API_KEY,
        "pageSize": 5,
    }

    async with httpx.AsyncClient() as client:
        r = await client.get(headlines_url, params=headlines_params, timeout=10)
        r.raise_for_status()
        data = r.json()

    articles = data.get("articles", [])
    if not articles:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        articles = data.get("articles", [])

    if not articles:
        return f"No articles found for query='{query}', country='{country}'"

    results = []
    for a in articles:
        results.append(
            f"Title: {a['title']}\n"
            f"Source: {a['source']['name']}\n"
            f"Published: {a['publishedAt']}\n"
            f"URL: {a['url']}\n"
        )
    return "\n---\n".join(results)


@mcp.tool()
async def analyze_sentiment(text: str) -> str:
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="You are a sentiment analyzer. Respond ONLY in this JSON format: {\"sentiment\": \"positive|negative|neutral\", \"confidence\": 0.0-1.0, \"reason\": \"one sentence\"}",
        messages=[{"role": "user", "content": text}],
    )
    return response.content[0].text

@mcp.tool()
async def search_documents(query: str) -> str:
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    embeddings = vo.embed(query, model = 'voyage-3', input_type = 'query').embeddings

    client = chromadb.CloudClient(
        api_key= CHROMA_API_KEY,
        database= CHROMA_DATABASE,
        tenant= CHROMA_TENANT
    )
    coll = client.get_collection('notes')
    results = coll.query(
        query_embeddings= embeddings,
        n_results= 3
    )
    result = ''
    for i,doc in enumerate(results['documents'][0],1):
        result += f"Document {i}:\n{doc}\n\n"
    return result

@mcp.tool()
async def get_weather(city: str) -> str:
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="You are a geographical map assistant. Get the latitude and longitude of given city up to 4 decimal places. Respond ONLY in this JSON format: {\"latitude\": 3.1412, \"longitude\": 101.6865}",
        messages=[{"role": "user", "content": city}],
    )

    cords = json.loads(response.content[0].text)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
	    "latitude": cords.get('latitude'),
	    "longitude": cords.get('longitude'),
	    "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min"],
	    "forecast_days": 1,
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(url, params= params ,timeout = 10)
        r.raise_for_status()
        data = r.json()
    result = f"WMO weather code: {data['daily']['weather_code'][0]}\nMax temperature: {data['daily']['temperature_2m_max'][0]}°C\n Min temperature: {data['daily']['temperature_2m_min'][0]}°C"
    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")