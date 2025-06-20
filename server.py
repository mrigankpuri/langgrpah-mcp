#!/usr/bin/env python
from fastmcp import FastMCP, Context
import asyncio

# 1. Create a FastMCP server
mcp = FastMCP(
    name="Weather Server",
)

# 2. Define a tool
@mcp.tool()
async def get_weather(city: str, ctx: Context):
    """Gets the weather for a given city."""
    # Send streaming progress updates and log messages
    await ctx.info(f"Starting weather fetch for {city}")
    await ctx.report_progress(progress=1, total=3, message="NOTIFICATION 1")
    
    await asyncio.sleep(1)
    
    await ctx.info(f"Fetching weather data for {city}...")
    await ctx.report_progress(progress=2, total=3, message="NOTIFICATION 2")
    
    await asyncio.sleep(1)
    
    await ctx.info(f"Weather data retrieved for {city}!")
    await ctx.report_progress(progress=3, total=3, message="NOTIFICATION 3")
    
    return f"It's always sunny in {city}."

# 3. Run the server
if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)