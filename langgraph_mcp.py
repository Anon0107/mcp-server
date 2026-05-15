import asyncio
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic
from dotenv import load_dotenv

load_dotenv()

SERVER_PYTHON = os.getenv("MCP_PYTHON", r"venv\Scripts\python")

class ResearchState(TypedDict):
    query: str
    weather: str
    news: str
    sentiment: str
    final_report: str

async def run_graph(query: str) -> str:
    server_params = StdioServerParameters(
        command=SERVER_PYTHON,
        args=["server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def weather_node(state: ResearchState) -> ResearchState:
                r = await session.call_tool("get_weather", {"city": "Kuala Lumpur"})
                return {"weather": r.content[0].text}

            async def news_node(state: ResearchState) -> ResearchState:
                r = await session.call_tool("search_news", {"query": state["query"], "country": "my"})
                return {"news": r.content[0].text}

            async def sentiment_node(state: ResearchState) -> ResearchState:
                news_snippet = state["news"][:1000]
                r = await session.call_tool("analyze_sentiment", {"text": news_snippet})
                return {"sentiment": r.content[0].text}

            async def report_node(state: ResearchState) -> ResearchState:
                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=500,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Synthesize this into a concise research report:\n\n"
                            f"Query: {state['query']}\n"
                            f"Weather: {state['weather']}\n"
                            f"News:\n{state['news'][:1500]}\n"
                            f"Sentiment: {state['sentiment']}\n\n"
                            f"Write 3-4 sentences max."
                        )
                    }]
                )
                return {"final_report": response.content[0].text}

            graph = StateGraph(ResearchState)
            graph.add_node("weather", weather_node)
            graph.add_node("news", news_node)
            graph.add_node("sentiment", sentiment_node)
            graph.add_node("report", report_node)

            graph.set_entry_point("weather")
            graph.add_edge("weather", "news")
            graph.add_edge("news", "sentiment")
            graph.add_edge("sentiment", "report")
            graph.add_edge("report", END)

            app = graph.compile()
            result = await app.ainvoke({
                "query": query,
                "weather": "",
                "news": "",
                "sentiment": "",
                "final_report": ""
            })

            return result["final_report"]

if __name__ == "__main__":
    report = asyncio.run(run_graph("AI startups in Malaysia"))
    print(report)