#!/usr/bin/env python
"""
Generic LangGraph Agent with FastMCP streaming client
Connect to any FastMCP server by providing the URL
"""
import asyncio
import argparse
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict, Annotated
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
import json

load_dotenv()

# State definition for LangGraph using the proper pattern
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Global configuration
class MCPConfig:
    def __init__(self, server_url: str, server_name: str = "Unknown Server"):
        self.server_url = server_url
        self.server_name = server_name
        self.client = None
        self.available_tools = []

# Global MCP configuration
mcp_config = None

async def initialize_mcp_client(server_url: str, server_name: str = "MCP Server"):
    """Initialize the global MCP client with any server URL"""
    global mcp_config
    mcp_config = MCPConfig(server_url, server_name)
    mcp_config.client = Client(server_url)
    await mcp_config.client.__aenter__()
    
    # Get available tools from the server
    tools_info = await mcp_config.client.list_tools()
    mcp_config.available_tools = [tool.name for tool in tools_info]
    
    return tools_info

@tool
async def call_mcp_tool_streaming(tool_name: str, tool_args: str) -> str:
    """
    Call any tool on the connected MCP server with streaming progress updates.
    
    Args:
        tool_name: Name of the tool to call on the MCP server
        tool_args: JSON string of arguments to pass to the tool
    """
    if not mcp_config or not mcp_config.client:
        return "Error: MCP client not initialized"
    
    if tool_name not in mcp_config.available_tools:
        return f"Error: Tool '{tool_name}' not available. Available tools: {', '.join(mcp_config.available_tools)}"
    
    try:
        # Parse arguments
        if tool_args.strip():
            args_dict = json.loads(tool_args)
        else:
            args_dict = {}
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON arguments: {e}"
    
    streaming_messages = []
    
    # Progress handler to capture streaming updates
    def progress_handler(progress_token, progress, total, message=None):
        if total is not None:
            progress_msg = f"🔧 Step {int(progress)}/{int(total)}: Executing {tool_name}..."
        else:
            progress_msg = f"🔧 Progress: {progress}"
        streaming_messages.append(progress_msg)
        print(f"📡 {progress_msg}")  # Real-time streaming output
    
    try:
        # Call the MCP tool with streaming
        result = await mcp_config.client.call_tool(
            tool_name,
            args_dict,
            progress_handler=progress_handler
        )
        
        # Extract the text content
        if result and len(result) > 0:
            tool_result = result[0].text
        else:
            tool_result = f"No result from {tool_name}"
        
        # Combine streaming messages with final result
        if streaming_messages:
            full_response = "\n".join([
                f"🔧 Tool Execution Progress for {tool_name}:",
                *streaming_messages,
                f"✅ Final Result: {tool_result}"
            ])
        else:
            full_response = tool_result
            
        return full_response
        
    except Exception as e:
        return f"❌ Error calling {tool_name}: {str(e)}"

def create_dynamic_tools():
    """Create dynamic tools based on available MCP tools"""
    # Use the generic call_mcp_tool_streaming for all tools
    # The LLM will decide which tool to call based on the tool_name parameter
    return [call_mcp_tool_streaming]

def chatbot_node(state: State) -> dict:
    """LLM node that decides which tools to call using bind_tools()"""
    # Get the LLM with tools bound
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"))
    
    # Create tools and bind them to the LLM
    tools = create_dynamic_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    # Add context about the connected server
    messages = state["messages"]
    if mcp_config:
        # Add system context if this is the first message or no system message exists
        has_system_message = any(hasattr(msg, 'type') and msg.type == 'system' for msg in messages)
        
        if not has_system_message and messages:
            context_message = f"""You are connected to {mcp_config.server_name} with streaming capabilities.

Available tools: {', '.join(mcp_config.available_tools)}

When users ask about tools or want to use them, call the 'call_mcp_tool_streaming' function with:
- tool_name: The name of the tool to call (e.g., 'get_weather')  
- tool_args: JSON string of arguments (e.g., '{{"city": "Tokyo"}}')

Always be helpful and explain what you're doing when calling tools."""
            
            # Insert system message at the beginning
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=context_message)] + messages
    
    # Invoke the LLM with bound tools
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Create the graph using the proper LangGraph pattern
def create_graph():
    """Create the LangGraph workflow"""
    # Create dynamic tools
    tools = create_dynamic_tools()
    
    # Build the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.set_entry_point("chatbot")
    
    return graph_builder.compile()

async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generic FastMCP Streaming Client')
    parser.add_argument('--url', required=True, help='MCP server URL (e.g., http://localhost:8001/mcp)')
    parser.add_argument('--name', default='MCP Server', help='Server name for display')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
    
    args = parser.parse_args()
    
    # Set model if provided
    if args.model:
        os.environ['OPENAI_MODEL'] = args.model
    
    print(f"🚀 Starting Generic MCP Streaming Client...")
    print(f"🌐 Connecting to: {args.url}")
    print(f"📛 Server name: {args.name}")
    
    # Initialize MCP client
    try:
        tools_info = await initialize_mcp_client(args.url, args.name)
        print("✅ Connected to MCP server!")
        print(f"📋 Available tools: {[tool.name for tool in tools_info]}")
        
        # Show tool descriptions
        print("\n🔧 Tool descriptions:")
        for tool in tools_info:
            print(f"  • {tool.name}: {tool.description}")
        
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")
        print("💡 Make sure the server is running and the URL is correct")
        return
    
    # Create the graph after MCP client is initialized
    app = create_graph()
    
    try:
        print(f"\n💡 How to use:")
        print(f"• Ask questions about available tools")
        print(f"• Request tool execution in natural language")
        print(f"• Example: 'Get weather for Tokyo'")
        print(f"• Example: 'What tools are available?'")
        print("• Type 'quit' to exit")
        print("-" * 80)
        
        while True:
            user_input = input(f"\n🧑 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            if not user_input:
                continue
            
            # Create initial state with user message
            initial_state = {"messages": [HumanMessage(content=user_input)]}
            
            print(f"\n🤖 Agent: ", end="", flush=True)
            
            # Run the agent
            final_state = await app.ainvoke(initial_state)
            
            # Print the final response
            if final_state["messages"]:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(last_message.content)
                else:
                    print("Response received!")
                
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Clean up MCP client
        if mcp_config and mcp_config.client:
            try:
                await mcp_config.client.__aexit__(None, None, None)
            except:
                pass

if __name__ == "__main__":
    asyncio.run(main()) 