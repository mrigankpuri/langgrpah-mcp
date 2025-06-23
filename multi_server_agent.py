#!/usr/bin/env python
"""
Multi-Server LangGraph Agent - Connects to multiple MCP servers
"""
import asyncio
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated, List, Any, Dict, Optional
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
import json
import contextlib

load_dotenv()

# Generic state definition for any LangGraph agent
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Configuration for multiple MCP servers
class MultiMCPConfig:
    def __init__(self):
        self.servers = {
            "rag": {
                "url": "http://localhost:8001/mcp",
                "name": "RAG Server",
                "tools": []
            },
            "summary": {
                "url": "http://localhost:8002/mcp", 
                "name": "Summarization Server",
                "tools": []
            }
        }
        self.all_tools = []

# Global MCP configuration
multi_mcp_config = MultiMCPConfig()

@contextlib.asynccontextmanager
async def get_mcp_client(server_url: str):
    """Generic context manager for any MCP client"""
    client = None
    try:
        client = Client(server_url)
        await client.__aenter__()
        yield client
    except Exception as e:
        print(f"⚠️ MCP client connection error for {server_url}: {e}")
        yield None
    finally:
        if client:
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                print(f"⚠️ MCP client cleanup error: {e}")

async def initialize_all_mcp_servers():
    """Initialize and discover tools from all MCP servers"""
    all_tools = []
    
    for server_key, server_config in multi_mcp_config.servers.items():
        print(f"🔍 Connecting to {server_config['name']}...")
        
        async with get_mcp_client(server_config["url"]) as client:
            if not client:
                print(f"❌ Could not connect to {server_config['name']}")
                continue
            
            try:
                tools_info = await client.list_tools()
                server_tools = [tool.name for tool in tools_info]
                server_config["tools"] = server_tools
                all_tools.extend(server_tools)
                
                print(f"✅ {server_config['name']}: {server_tools}")
                
            except Exception as e:
                print(f"❌ Error discovering tools from {server_config['name']}: {e}")
    
    multi_mcp_config.all_tools = all_tools
    print(f"\n🛠️ Total available tools: {all_tools}")
    return all_tools

def find_server_for_tool(tool_name: str) -> Optional[Dict]:
    """Find which server has the specified tool"""
    for server_key, server_config in multi_mcp_config.servers.items():
        if tool_name in server_config["tools"]:
            return server_config
    return None

@tool
async def call_mcp_tool_streaming(tool_name: str, tool_args: str) -> str:
    """
    Generic tool caller that works with multiple MCP servers.
    
    Args:
        tool_name: Name of the tool to call
        tool_args: JSON string of arguments to pass to the tool
    """
    # Find which server has this tool
    server_config = find_server_for_tool(tool_name)
    if not server_config:
        return f"❌ Tool '{tool_name}' not found. Available tools: {', '.join(multi_mcp_config.all_tools)}"
    
    try:
        # Parse arguments
        if tool_args.strip():
            args_dict = json.loads(tool_args)
        else:
            args_dict = {}
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON arguments: {e}"
    
    # Collect streaming messages from the tool itself
    streaming_messages = []
    
    # Minimal progress handler - just capture what the tool sends
    def progress_handler(progress_token, progress, total, message=None):
        if message:
            # Use the actual message from the tool
            streaming_messages.append(f"📡 {message}")
            print(f"📡 {message}")
        elif total is not None and progress is not None:
            # Minimal generic progress if no message
            streaming_messages.append(f"📡 {tool_name}: {int(progress)}/{int(total)}")
            print(f"📡 {tool_name}: {int(progress)}/{int(total)}")
    
    print(f"🔧 Calling {tool_name} on {server_config['name']}...")
    
    async with get_mcp_client(server_config["url"]) as client:
        if not client:
            return f"❌ Error: Could not connect to {server_config['name']}"
        
        try:
            result = await client.call_tool(
                tool_name,
                args_dict,
                progress_handler=progress_handler
            )
            
            tool_result = result[0].text if result and len(result) > 0 else f"No result from {tool_name}"
            
            # Just return the tool result with any streaming messages it provided
            if streaming_messages:
                return f"{tool_result}\n\n📡 Progress: {len(streaming_messages)} updates received from {server_config['name']}"
            else:
                return tool_result
                
        except Exception as e:
            return f"❌ Error calling {tool_name} on {server_config['name']}: {str(e)}"

def create_dynamic_tools():
    """Create dynamic tools based on available MCP tools"""
    return [call_mcp_tool_streaming]

def chatbot_node(state: AgentState) -> dict:
    """Generic LLM node that decides which tools to call using bind_tools()"""
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"))
    
    # Create tools and bind them to the LLM
    tools = create_dynamic_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    # Add context about available tools
    messages = state["messages"]
    
    # Add system context if this is the first message or no system message exists
    has_system_message = any(hasattr(msg, 'type') and getattr(msg, 'type', None) == 'system' for msg in messages)
    
    if not has_system_message and messages and multi_mcp_config.all_tools:
        # Create server-specific tool descriptions
        server_descriptions = []
        for server_key, server_config in multi_mcp_config.servers.items():
            if server_config["tools"]:
                server_descriptions.append(f"• {server_config['name']}: {', '.join(server_config['tools'])}")
        
        context_message = f"""You are a helpful AI assistant connected to multiple MCP servers with streaming capabilities.

Available servers and tools:
{chr(10).join(server_descriptions)}

When users ask questions that require tool usage, call the 'call_mcp_tool_streaming' function with:
- tool_name: The name of the tool to call (I'll automatically find the right server)
- tool_args: JSON string of arguments for the tool

Tool usage examples:
- For evidence/research: Use 'rag_tool' with {{"query": "your search query"}}
- For summaries/reports: Use 'summarization_tool' with {{"product": "product name", "output_format": "markdown"}}

Always be helpful and explain what you're doing when calling tools."""
        
        # Insert system message at the beginning
        messages = [SystemMessage(content=context_message)] + messages
    
    # Invoke the LLM with bound tools
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def create_multi_server_graph():
    """Create a LangGraph workflow that works with multiple MCP servers"""
    # Create dynamic tools
    tools = create_dynamic_tools()
    
    # Build the graph using the proper LangGraph pattern
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.set_entry_point("chatbot")
    
    return graph_builder.compile()

async def process_query(user_query: str) -> dict:
    """
    Process query using multiple MCP servers
    
    Args:
        user_query: The user's question or request
    """
    # Initialize all MCP servers
    await initialize_all_mcp_servers()
    
    # Create the graph
    app = create_multi_server_graph()
    
    # Create initial state with user message
    initial_state = {"messages": [HumanMessage(content=user_query)]}
    
    try:
        # Run the agent
        final_state = await app.ainvoke(initial_state)
        
        # Return the final state with metadata
        result = {
            "messages": final_state["messages"],
            "query": user_query,
            "servers": {k: v["name"] for k, v in multi_mcp_config.servers.items()},
            "available_tools": multi_mcp_config.all_tools
        }
        
        return result
        
    except Exception as e:
        # Return error state
        return {
            "messages": [AIMessage(content=f"❌ Error processing query: {str(e)}")],
            "query": user_query,
            "error": str(e),
            "servers": {k: v["name"] for k, v in multi_mcp_config.servers.items()},
            "available_tools": multi_mcp_config.all_tools
        }

async def main():
    """Interactive main function for testing"""
    print("🚀 Starting Multi-Server LangGraph Agent...")
    print("🌐 Connecting to multiple MCP servers...")
    
    # Initialize all servers
    tools = await initialize_all_mcp_servers()
    if not tools:
        print("❌ No tools available. Make sure MCP servers are running.")
        return
    
    print("\n💡 How to use:")
    print("• Ask questions that require any of the available tools")
    print("• Example: 'Find evidence about API performance' (uses RAG server)")
    print("• Example: 'Create a summary for analytics platform' (uses Summary server)")
    print("• Type 'quit' to exit")
    print("-" * 80)
    
    try:
        while True:
            user_input = input(f"\n🧑 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            if not user_input:
                continue
            
            print(f"\n🤖 Agent: ", end="", flush=True)
            
            # Process the query
            result = await process_query(user_input)
            
            # Print the final response
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    print(last_message.content)
                else:
                    print("Response received!")
                
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 