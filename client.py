#!/usr/bin/env python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    # 1. Create a client
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "streamable_http",
                "url": "http://localhost:8001/mcp/",
            },
        }
    )

    # 2. Get tools from the server
    tools = await client.get_tools()
    print(f"Got tools: {[tool.name for tool in tools]}")

    # 3. Create a langgraph agent
    # Make sure to configure your OpenAI API key either in your environment
    # or directly here.
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4"))
    agent = create_react_agent(llm, tools)

    # 4. Invoke the agent
    query = "What is the weather in San Francisco?"
    print(f"--- Asking: {query} ---")
    response = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

    print("--- Agent Response ---")
    print(response['messages'][-1].content)
    print("----------------------")


if __name__ == "__main__":
    asyncio.run(main())