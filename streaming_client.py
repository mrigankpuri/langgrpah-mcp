#!/usr/bin/env python
"""
LangGraph Agent with FastMCP streaming client
"""
import asyncio
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import List, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()

# State definition for LangGraph
class AgentState(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    streaming_messages: List[str] = Field(default_factory=list)

# Global FastMCP client
mcp_client = None

async def initialize_mcp_client():
    """Initialize the global MCP client"""
    global mcp_client
    mcp_client = Client("http://localhost:8001/mcp")
    await mcp_client.__aenter__()

@tool
async def get_weather_streaming(city: str) -> str:
    """
    Get weather for a city with streaming progress updates.
    
    Args:
        city: City name to get weather for
    """
    if not mcp_client:
        return "Error: MCP client not initialized"
    
    streaming_messages = []
    
    # Progress handler to capture streaming updates
    def progress_handler(progress_token, progress, total, message=None):
        if total is not None:
            progress_msg = f"🌤️ Step {int(progress)}/{int(total)}: Processing weather data..."
        else:
            progress_msg = f"🌤️ Progress: {progress}"
        streaming_messages.append(progress_msg)
        print(f"📡 {progress_msg}")  # Real-time streaming output
    
    try:
        # Call the MCP tool with streaming
        result = await mcp_client.call_tool(
            "get_weather",
            {"city": city},
            progress_handler=progress_handler
        )
        
        # Extract the text content
        if result and len(result) > 0:
            weather_result = result[0].text
        else:
            weather_result = f"No weather data available for {city}"
        
        # Combine streaming messages with final result
        if streaming_messages:
            full_response = "\n".join([
                f"🌤️ Weather Fetch Progress for {city}:",
                *streaming_messages,
                f"✅ Final Result: {weather_result}"
            ])
        else:
            full_response = weather_result
            
        return full_response
        
    except Exception as e:
        return f"❌ Error getting weather for {city}: {str(e)}"

# Create tool node
tools = [get_weather_streaming]
tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end"""
    messages = state.messages
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

async def call_model(state: AgentState) -> dict:
    """Model that decides when to use streaming tools"""
    messages = state.messages
    last_message = messages[-1] if messages else None
    
    # Initialize the LLM
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"))
    
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        
        # Check if user is asking about weather
        if any(word in content for word in ["weather", "temperature", "forecast", "sunny", "rain", "cloudy"]):
            # Extract city name (simple approach)
            city = "San Francisco"  # default
            
            # Try to extract city from message
            words = content.split()
            for i, word in enumerate(words):
                if word.lower() in ["in", "for", "at"] and i + 1 < len(words):
                    city = words[i + 1].title()
                    break
            
            response = AIMessage(
                content=f"I'll get the weather for {city} with real-time streaming updates.",
                tool_calls=[{
                    "name": "get_weather_streaming",
                    "args": {"city": city},
                    "id": "weather_call_1"
                }]
            )
        else:
            # Use LLM for general conversation
            try:
                ai_response = await llm.ainvoke(messages)
                response = AIMessage(
                    content=ai_response.content + "\n\n💡 Ask me about weather in any city to see streaming updates!"
                )
            except Exception as e:
                response = AIMessage(
                    content=f"""Hello! I can help you get weather information with real-time streaming updates:

• Ask "What's the weather in [city]?" to get streaming weather data
• Try: "What's the weather in New York?"
• Or: "Get weather for London"

I'll show you real-time progress as I fetch the data!

Error connecting to LLM: {str(e)}"""
                )
    elif isinstance(last_message, ToolMessage):
        response = AIMessage(
            content=f"Here's your weather information:\n\n{last_message.content}"
        )
    else:
        response = AIMessage(
            content="I'm ready to help with streaming weather information!"
        )
    
    return {"messages": state.messages + [response]}

# Create the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.set_entry_point("agent")

# Compile the graph
app = workflow.compile()

async def main():
    """Main execution function"""
    print("🚀 Starting LangGraph Agent with FastMCP Streaming...")
    
    # Initialize MCP client
    try:
        await initialize_mcp_client()
        print("✅ Connected to FastMCP server!")
        
        # Test connection
        tools_info = await mcp_client.list_tools()
        print(f"📋 Available tools: {[tool.name for tool in tools_info]}")
        
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")
        print("💡 Make sure the server is running: python server.py")
        return
    
    try:
        print("\n💡 Try these commands:")
        print("• 'What's the weather in Paris?' - Get streaming weather data")
        print("• 'Weather for Tokyo' - Another city example")  
        print("• 'Hello' - General conversation")
        print("• 'quit' - Exit")
        print("-" * 60)
        
        while True:
            user_input = input("\n🧑 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            if not user_input:
                continue
            
            # Create initial state
            initial_state = AgentState(messages=[HumanMessage(content=user_input)])
            
            print("\n🤖 Agent: ", end="", flush=True)
            
            # Run the agent
            agent_responded = False
            async for output in app.astream(initial_state):
                for key, value in output.items():
                    if key == "agent" and "messages" in value:
                        last_message = value["messages"][-1]
                        if isinstance(last_message, AIMessage):
                            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                                print(last_message.content)
                                agent_responded = True
                    elif key == "tools" and "messages" in value:
                        # Tool execution shows streaming progress in real-time
                        pass
            
            if not agent_responded:
                print("Processing complete!")
                
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Clean up MCP client
        if mcp_client:
            try:
                await mcp_client.__aexit__(None, None, None)
            except:
                pass

if __name__ == "__main__":
    asyncio.run(main()) 