#!/usr/bin/env python
"""
🐝 Beekeeper Agent - LangGraph with Intent Detection Business Flow
Restored original business flow: Intent Detection → Evidence Discovery OR Claim Generation → Tools
"""
import asyncio
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Annotated, List, Any, Dict, Optional, Literal
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
import json
import contextlib
from langgraph.prebuilt import ToolNode
from uuid import uuid4
from langgraph.config import get_stream_writer

load_dotenv()

# Beekeeper State with intent detection and proper typing
class BeekeeperState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[Literal["evidence_discovery", "claim_generation", "clarification_needed"]]
    selected_server: Optional[Literal["rag_server", "summary_server"]]
    conversation_complete: Optional[bool]  # True when conversational response is provided

# MCP Server Configurations Registry
SERVER_CONFIGS = {
    "rag_server": {
        "url": "http://localhost:8001/mcp",
        "name": "RAG Server", 
        "purpose": "Evidence Discovery"
    },
    "summary_server": {
        "url": "http://localhost:8002/mcp",
        "name": "Summarization Server",
        "purpose": "Claim Generation"
    }
}

def get_available_servers() -> list[str]:
    """Get list of available server keys"""
    return list(SERVER_CONFIGS.keys())

@contextlib.asynccontextmanager
async def get_mcp_client(server_key: str):
    """Get MCP client for specific server"""
    client = None
    try:
        if server_key not in SERVER_CONFIGS:
            raise ValueError(f"Unknown server key: {server_key}. Available: {list(SERVER_CONFIGS.keys())}")
            
        server_info = SERVER_CONFIGS[server_key]
        client = Client(server_info["url"])
        await client.__aenter__()
        yield client, server_info
    except Exception as e:
        print(f"⚠️ {server_key} connection error: {e}")
        yield None, None
    finally:
        if client:
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                print(f"⚠️ {server_key} cleanup error: {e}")

# Intent Detection Node (handles initial queries AND post-tool responses)
def intent_detection_node(state: BeekeeperState) -> dict:
    """🎯 Intent Detection: Handle initial queries OR provide conversational responses after tools"""
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4"))
    
    messages = state["messages"]
    
    # Handle empty messages
    if not messages:
        return {
            "intent": "evidence_discovery", 
            "selected_server": "rag_server",
            "messages": [AIMessage(content="Hello! I'm your Beekeeper assistant. I can help you find evidence or generate summaries. What would you like me to help you with?")]
        }
    
    # Use LLM to determine the conversation state and what to do next
    conversation_analysis_prompt = SystemMessage(content="""You are analyzing a conversation to determine what action to take next. 

Look at the conversation and determine:
1. Is this an initial user query that needs to be processed?
2. Has the user's query been answered with tool results that need a conversational response?
3. Does the user need clarification?

Respond in JSON format:
{
  "conversation_state": "initial_query" | "query_answered" | "needs_clarification",
  "intent": "evidence_discovery" | "claim_generation" | "clarification_needed" (only for initial_query),
  "response": "your conversational response"
}

For initial_query classification:
- EVIDENCE_DISCOVERY: finding/searching/researching/analyzing existing information
  Examples: "Find evidence about...", "Search for...", "Research...", "What are the...", "Show me data..."
  
- CLAIM_GENERATION: creating/generating/writing new content, claims, summaries, reports
  Examples: "Generate claims about...", "Create a summary...", "Write a report...", "Make claims...", "Produce..."
  
For query_answered: If you see tool results (like search results or generated summaries), provide a conversational response about those results.

For needs_clarification: If the request is vague or ambiguous, ask for clarification.""")
    
    # Create conversation context for analysis
    conversation_text = "\n".join([
        f"{'User' if not isinstance(msg, AIMessage) else 'Assistant'}: {msg.content if hasattr(msg, 'content') else str(msg)}"
        for msg in messages[-5:]  # Last 5 messages for context
    ])
    
    analysis_messages = [
        conversation_analysis_prompt,
        HumanMessage(content=f"Analyze this conversation:\n{conversation_text}")
    ]
    
    try:
        # Get LLM analysis
        response = llm.invoke(analysis_messages)
        result = json.loads(response.content.strip())
        
        conversation_state = result.get("conversation_state", "initial_query")
        intent = result.get("intent")
        conversational_response = result.get("response", "I'll help you with that!")
        
        print(f"Conversation Analysis: {conversation_state}, intent: {intent}")
        
        # Handle based on conversation state
        if conversation_state == "query_answered":
            # Query has been answered, provide conversational response and end
            return {
                "messages": [AIMessage(content=conversational_response)],
                "conversation_complete": True
            }
            
        elif conversation_state == "needs_clarification":
            # Needs clarification, provide response and end
            return {
                "intent": "clarification_needed",
                "selected_server": None,
                "messages": [AIMessage(content=conversational_response)]
            }
            
        else:  # initial_query
            # Initial query, classify intent and route to business logic
            if intent == "claim_generation":
                return {
                    "intent": "claim_generation", 
                    "selected_server": "summary_server",
                    "messages": [AIMessage(content=conversational_response)]
                }
            else:
                # Default to evidence_discovery
                return {
                    "intent": "evidence_discovery", 
                    "selected_server": "rag_server",
                    "messages": [AIMessage(content=conversational_response)]
                }
        
    except Exception as e:
        print(f"⚠️ LLM conversation analysis error: {e}")
        # On error, provide default friendly response
        last_message = messages[-1] if messages else None
        user_query = last_message.content if last_message and hasattr(last_message, 'content') else "your request"
        
        clarification_message = f"""Hi there! I'd be happy to help you with: "{user_query}"

I specialize in two main areas:

Finding Information - I can search for evidence, research data, analyze existing information
Creating Content - I can generate summaries, write reports, create documentation

Could you help me understand - are you looking for me to **find existing information** or **create new content** for you?"""

        return {
            "intent": "clarification_needed",
            "selected_server": None,
            "messages": [AIMessage(content=clarification_message)]
        }

# Evidence Discovery Node (simplified - only handles tool calling)
def evidence_discovery_node(state: BeekeeperState) -> dict:
    """🔍 Evidence Discovery: Create tool calls for RAG server"""
    print("🔍 Evidence Discovery: Preparing to search for information...")
    
    messages = state["messages"]
    if not messages:
        return {"messages": []}
    
    # Get the user's query from the most recent human message
    user_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    if not user_query:
        return {"messages": []}
    
    # Create tool call for RAG server
    tool_call = {
        "name": "call_mcp_tool",
        "args": {
            "server_key": "rag_server",
            "tool_name": "rag_tool", 
            "tool_args": {"query": user_query}
        },
        "id": f"call_{uuid4()}"
    }
    
    # Create AI message with tool call
    ai_message = AIMessage(content="Searching for evidence...", tool_calls=[tool_call])
    
    return {"messages": [ai_message]}

# Claim Generation Node (simplified - only handles tool calling) 
def claim_generation_node(state: BeekeeperState) -> dict:
    """📝 Claim Generation: Create tool calls for summarization server"""
    print("📝 Claim Generation: Preparing to generate content...")
    
    messages = state["messages"]
    if not messages:
        return {"messages": []}
    
    # Get the user's query from the most recent human message
    user_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    if not user_query:
        return {"messages": []}
    
    # Check if query contains product information for claim generation
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4"))
    
    product_detection_prompt = SystemMessage(content="""You are analyzing a user query to determine if it contains information about a specific product or topic for generating claims/summaries.

Respond in JSON format:
{
  "has_product": true/false,
  "product_name": "extracted product name" (if found),
  "confidence": 0.0-1.0
}

Look for:
- Explicit product mentions ("claims about iPhone", "summary for Tesla Model 3")
- Topic specifications ("generate claims for renewable energy", "summarize AI trends")
- Subject matter ("create summary about blockchain technology")

If no specific product/topic is mentioned, set has_product to false.""")
    
    try:
        detection_response = llm.invoke([
            product_detection_prompt,
            HumanMessage(content=f"Analyze this query for product/topic information: {user_query}")
        ])
        
        detection_result = json.loads(detection_response.content.strip())
        has_product = detection_result.get("has_product", False)
        product_name = detection_result.get("product_name", "")
        confidence = detection_result.get("confidence", 0.0)
        
        print(f"📝 Product Detection: has_product={has_product}, product={product_name}, confidence={confidence}")
        
        if not has_product or confidence < 0.7:
            # Return clarification request
            clarification_message = f"""📝 I'd be happy to generate claims or summaries for you! However, I need to know what specific product or topic you'd like me to focus on.

For example, you could ask:
• "Generate claims about the iPhone 15"
• "Create a summary for Tesla's latest earnings"
• "Write claims about renewable energy benefits"
• "Summarize trends in artificial intelligence"

Could you please specify what product or topic you'd like me to generate content about?"""
            
            return {
                "intent": "clarification_needed",
                "messages": [AIMessage(content=clarification_message)]
            }
            
    except Exception as e:
        print(f"⚠️ Product detection error: {e}")
        # On error, ask for clarification
        clarification_message = """📝 I'd be happy to help generate claims or summaries! Could you please specify what product or topic you'd like me to focus on?

For example: "Generate claims about [product name]" or "Create a summary for [topic]"."""
        
        return {
            "intent": "clarification_needed", 
            "messages": [AIMessage(content=clarification_message)]
        }
    
    # Create tool call for summarization server
    tool_call = {
        "name": "call_mcp_tool",
        "args": {
            "server_key": "summary_server",
            "tool_name": "summarization_tool",
            "tool_args": {"product": product_name or user_query}
        },
        "id": f"call_{uuid4()}"
    }
    
    # Create AI message with tool call
    ai_message = AIMessage(content="Generating content...", tool_calls=[tool_call])
    
    return {"messages": [ai_message]}

# Generic MCP Tool Calling Function
@tool
async def call_mcp_tool(server_key: str, tool_name: str, tool_args: dict) -> str:
    """
    Generic function to call any tool on any MCP server with real-time LangGraph streaming
    
    Args:
        server_key: The server configuration key (available: rag_server, summary_server)
        tool_name: Name of the tool to call
        tool_args: Dictionary of arguments to pass to the tool
    """
    from langgraph.config import get_stream_writer
    
    # Get LangGraph stream writer for real-time custom streaming
    try:
        writer = get_stream_writer()
    except Exception as e:
        print(f"⚠️ Could not get stream writer: {e}")
        writer = None
    
    # Simple progress handler for ctx.report_progress() calls
    def progress_handler(progress_token, progress, total, message=None):
        """Handle MCP progress notifications - FastMCP puts message in 'total' parameter"""
        if writer and isinstance(total, str):
            writer({"mcp_notification": total, "type": "mcp_progress"})
    
    async with get_mcp_client(server_key) as (client, server_info):
        if not client:
            error_msg = f"Could not connect to {server_info['name'] if server_info else server_key}"
            if writer:
                writer({"mcp_notification": error_msg, "type": "tool_error"})
            return error_msg
        
        try:
            print(f"Calling {tool_name} on {server_info['name']}...")
            
            # LANGGRAPH CUSTOM STREAMING: Notify start of tool execution
            if writer:
                writer({"mcp_notification": f"Calling {tool_name} on {server_info['name']}...", "type": "tool_start"})
            
            # Simple call with just progress handler - no event capture needed
            result = await client.call_tool(tool_name, tool_args, progress_handler=progress_handler)
            
            tool_result = result[0].text if result and len(result) > 0 else f"No result from {tool_name}"
            
            # LANGGRAPH CUSTOM STREAMING: Yield completion
            if writer:
                writer({"mcp_notification": f"Tool completed successfully", "type": "tool_complete"})
            
            # Just return the clean result
            return tool_result
                
        except Exception as e:
            print(f"Error details: {e}")
            error_msg = f"Error calling {tool_name} on {server_info['name'] if server_info else server_key}: {str(e)}"
            
            # LANGGRAPH CUSTOM STREAMING: Yield error immediately
            if writer:
                writer({"mcp_notification": error_msg, "type": "tool_error"})
            
            return error_msg

# Conditional Routing Logic (updated for cleaner flow)
def route_after_intent_detection(state: BeekeeperState) -> str:
    """Route after intent detection based on intent and conversation state"""
    intent = state.get("intent")
    conversation_complete = state.get("conversation_complete", False)
    
    print(f"🔄 Routing after intent detection: intent={intent}, complete={conversation_complete}")
    
    # If conversation is complete, end the flow
    if conversation_complete:
        return "END"
    
    # Route based on detected intent
    if intent == "evidence_discovery":
        return "evidence_discovery"
    elif intent == "claim_generation":
        return "claim_generation"
    else:  # clarification_needed or unknown
        return "END"


def route_after_evidence_discovery(state: BeekeeperState) -> str:
    """Route after evidence discovery - check if we need clarification or should call tools"""
    messages = state.get("messages", [])
    intent = state.get("intent")
    
    # If intent changed to clarification_needed, end the flow
    if intent == "clarification_needed":
        return "END"
    
    # Check if the last message has tool calls
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "rag_tool"
    
    return "END"


def route_after_claim_generation(state: BeekeeperState) -> str:
    """Route after claim generation - check if we need clarification or should call tools"""
    messages = state.get("messages", [])
    intent = state.get("intent")
    
    # If intent changed to clarification_needed, end the flow
    if intent == "clarification_needed":
        return "END"
        
    # Check if the last message has tool calls
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "summary_tool"
    
    return "END"

# Create the Beekeeper Agent Graph (simplified with standard ToolNode)
def create_beekeeper_agent():
    """🏗️ Create the Beekeeper Agent with LangGraph streaming support"""
    print("Creating Beekeeper Agent with LangGraph streaming support...")
    
    # Create graph
    workflow = StateGraph(BeekeeperState)
    
    # Add nodes
    workflow.add_node("intent_detection", intent_detection_node)
    workflow.add_node("evidence_discovery", evidence_discovery_node)
    workflow.add_node("claim_generation", claim_generation_node)
    
    # Add standard LangGraph tool nodes - streaming happens automatically via get_stream_writer()
    workflow.add_node("rag_tool", ToolNode([call_mcp_tool]))
    workflow.add_node("summary_tool", ToolNode([call_mcp_tool]))
    
    # Set entry point
    workflow.set_entry_point("intent_detection")
    
    # Add conditional routing from intent detection
    workflow.add_conditional_edges(
        "intent_detection",
        route_after_intent_detection,
        {
            "evidence_discovery": "evidence_discovery",
            "claim_generation": "claim_generation", 
            "END": END
        }
    )
    
    # Add conditional routing from evidence discovery
    workflow.add_conditional_edges(
        "evidence_discovery",
        route_after_evidence_discovery,
        {
            "rag_tool": "rag_tool",
            "END": END
        }
    )
    
    # Add conditional routing from claim generation
    workflow.add_conditional_edges(
        "claim_generation",
        route_after_claim_generation,
        {
            "summary_tool": "summary_tool",
            "END": END
        }
    )
    
    # Add edges from tool nodes back to intent detection
    workflow.add_edge("rag_tool", "intent_detection")
    workflow.add_edge("summary_tool", "intent_detection")
    
    # Compile the graph
    app = workflow.compile()
    
    print("Beekeeper Agent created successfully with LangGraph streaming support!")
    return app

async def process_query(user_query: str) -> dict:
    """
    Process a user query through the Beekeeper Agent with cleaner architecture
    
    Flow: Intent Detection → Business Logic → Tools → Intent Detection → END
    """
    # Create the Beekeeper graph
    app = create_beekeeper_agent()
    
    # Create initial state with user message
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "intent": None,
        "selected_server": None,
        "conversation_complete": False
    }
    
    try:
        print(f"Processing query: {user_query}")
        
        # Stream through the graph
        final_state = None
        async for state in app.astream(initial_state):
            print(f"State update: {list(state.keys())}")
            final_state = state
        
        # Extract final messages
        if final_state:
            # Get the state from the last node that executed
            last_node_state = list(final_state.values())[-1]
            messages = last_node_state.get("messages", [])
            
            # Get the final AI response
            final_response = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                    # Skip tool call messages
                    if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                        final_response = msg.content
                        break
            
            return {
                "response": final_response or "I apologize, I couldn't process your request properly.",
                "intent": last_node_state.get("intent"),
                "selected_server": last_node_state.get("selected_server"),
                "success": True
            }
        else:
            return {
                "response": "I apologize, I couldn't process your request.",
                "intent": None,
                "selected_server": None,
                "success": False
            }
        
    except Exception as e:
        print(f"\nError: {e}")
        return {
            "response": f"I encountered an error: {str(e)}",
            "intent": None,
            "selected_server": None,
            "success": False
        }

async def test_server_connections():
    """Test connections to all configured MCP servers"""
    print("Testing MCP Server Connections...")
    
    for server_key in get_available_servers():
        async with get_mcp_client(server_key) as (client, server_info):
            if client and server_info:
                try:
                    tools = await client.list_tools()
                    print(f"{server_info['name']}: {[t.name for t in tools]}")
                except Exception as e:
                    print(f"{server_info['name']}: {e}")
            else:
                server_config = SERVER_CONFIGS[server_key]
                print(f"{server_config['name']}: Connection failed")

async def main():
    """Interactive main function for testing Beekeeper business flow"""
    print("Starting Beekeeper Agent with Intent Detection Business Flow...")
    print("Architecture: Conversational Intent Detection → Business Logic → Tools → Conversational Response")
    print("Simplified: Intent detection is conversational, business logic handles tool responses")
    
    # Test server connections
    await test_server_connections()
    
    print(f"\nConversational Business Flow:")
    print(f"• Intent Detection: conversational acknowledgment of what will be done")
    print(f"• Evidence Discovery: tool call → conversational response to results")
    print(f"• Claim Generation: tool call → conversational response to results")
    print(f"• Generic MCP Tool: Single function handles all server/tool combinations")
    print(f"• Natural Flow: Each step feels like talking to a helpful assistant")
    
    print("\nExample queries:")
    print("• 'Find evidence about API performance' → Evidence Discovery → RAG ToolNode → Chat Response")
    print("• 'Create a summary for Analytics Platform' → Claim Generation → Summary ToolNode → Chat Response")
    print("• 'Generate a report about Customer Portal' → Claim Generation → Summary ToolNode → Chat Response")
    print("• 'Write a summary of our CRM system' → Claim Generation → Summary ToolNode → Chat Response")
    print("• 'Create a summary' (no product) → LLM detects missing product → Asks for clarification")
    print("• 'Help me with the project' → Intent Detection provides clarification directly")
    print("• Type 'quit' to exit")
    print("-" * 80)
    
    try:
        while True:
            user_input = input(f"\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            if not user_input:
                continue
            
            print(f"\nBeekeeper: ", end="", flush=True)
            
            # Process the query through business flow
            result = await process_query(user_input)
            
            # Print the final response
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    print(last_message.content)
                else:
                    print("Response received!")
            
            # Show business flow info
            intent = result.get("intent")
            server = result.get("selected_server")
            if intent and server:
                print(f"\nFlow: {intent} → {server}")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 