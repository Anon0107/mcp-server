# mcp-server

A personal MCP (Model Context Protocol) server exposing 4 tools via FastMCP, connectable to Claude Desktop and any MCP-compatible client.

## Tools

| Tool | Description |
|---|---|
| `search_news(query, country)` | Fetches top headlines via NewsAPI |
| `analyze_sentiment(text)` | Returns sentiment + confidence via Claude Haiku |
| `search_documents(query)` | Semantic search over specified collection in ChromaDB via Voyage AI embeddings |
| `get_weather(city)` | 1-day weather forecast via Open-Meteo API |

## Stack

- [FastMCP](https://github.com/modelcontextprotocol/python-sdk) — MCP server framework
- Anthropic Claude Haiku — sentiment analysis + geocoding
- Voyage AI `voyage-3` — document embeddings
- Chroma Cloud — vector database
- NewsAPI — news headlines
- Open-Meteo — free weather API

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
```

## Run locally

```bash
python server.py
```

Test with MCP Inspector:
```bash
mcp dev server.py
```

## Connect to Claude Desktop

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "kai-tools": {
      "command": "C:\\absolute\\path\\to\\.venv\\Scripts\\python.exe",
      "args": ["C:\\absolute\\path\\to\\server.py"],
      "env": {
        "NEWS_API_KEY": "...",
        "ANTHROPIC_API_KEY": "...",
        "VOYAGE_API_KEY": "...",
        "CHROMA_API_KEY": "...",
        "CHROMA_DATABASE": "...",
        "CHROMA_TENANT": "..."
      }
    }
  }
}
```