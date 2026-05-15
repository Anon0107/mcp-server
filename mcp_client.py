import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PYTHON = os.getenv("MCP_PYTHON", r"venv\Scripts\python")

async def main():
    server_params = StdioServerParameters(
        command=SERVER_PYTHON,
        args=["server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("=== Available Tools ===")
            for tool in tools.tools:
                print(f"  {tool.name}: {tool.description}")

            print("\n=== get_weather ===")
            r = await session.call_tool("get_weather", {"city": "Kuala Lumpur"})
            print(r.content[0].text)

            print("\n=== search_news ===")
            r = await session.call_tool("search_news", {"query": "AI Malaysia", "country": "my"})
            print(r.content[0].text[:500])

            print("\n=== analyze_sentiment ===")
            r = await session.call_tool("analyze_sentiment", {"text": "The economy is looking strong this quarter."})
            print(r.content[0].text)

            print("\n=== search_documents ===")
            r = await session.call_tool("search_documents", {"query": "machine learning"})
            print(r.content[0].text[:500])

asyncio.run(main())