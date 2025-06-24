#!/usr/bin/env python

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

# Chat Node - Handle conversation and user responses
def chat_node(state: BeekeeperState) -> dict:
    """💬 Chat Node: Handle conversational responses and user interaction"""
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4"))
    
    messages = state["messages"]
    
    # Handle empty messages
    if not messages:
        return {
            "messages": [AIMessage(content="Hello! I'm your Boston Scientific marketing assistant. I can help you find evidence for existing claims or generate new marketing claims. What would you like me to help you with?")]
        }
    
    # Check if we have tool results to respond to
    last_message = messages[-1] if messages else None
    
    # If last message is a tool result, provide conversational response
    if isinstance(last_message, ToolMessage):
        conversation_prompt = SystemMessage(content="""You are a helpful assistant for Boston Scientific's marketing team. 
        
You just received results from either:
1. Evidence Discovery: Research results supporting marketing claims
2. Marketing Claim Generation: New marketing content for products

Provide a friendly, professional conversational response about the results. Be concise but helpful.
Keep the response focused on how this helps their marketing efforts.""")
        
        try:
            response = llm.invoke([
                conversation_prompt,
                HumanMessage(content=f"Tool result: {last_message.content}")
            ])
            
            return {
                "messages": [AIMessage(content=response.content)],
                "conversation_complete": True
            }
            
        except Exception as e:
            print(f"⚠️ Chat response error: {e}")
            return {
                "messages": [AIMessage(content="I've completed your request. Is there anything else I can help you with?")],
                "conversation_complete": True
            }
    
    # For initial queries, provide acknowledgment and route to intent detection
    return {"messages": []}

# Intent Detection Node - Classify workflows
def intent_detection_node(state: BeekeeperState) -> dict:
    """🎯 Intent Detection: Classify evidence_discovery vs marketing_claim workflows"""
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4"))
    
    messages = state["messages"]
    
    # Handle empty messages
    if not messages:
        return {
            "intent": "clarification_needed",
            "messages": [AIMessage(content="Hello! I'm your Boston Scientific marketing assistant. How can I help you today?")]
        }
    
    # Get the user's query from the most recent human message
    user_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    if not user_query:
        return {
            "intent": "clarification_needed",
            "messages": [AIMessage(content="I'd be happy to help! Could you please tell me what you need assistance with?")]
        }
    
    # LLM-based intent classification
    intent_prompt = SystemMessage(content="""You are analyzing queries for Boston Scientific's marketing team to classify into two workflows:

1. EVIDENCE_DISCOVERY: Finding evidence to support existing marketing claims, regulatory requirements, or competitive analysis
   - Keywords: find, search, evidence, support, validate, verify, research, data, studies, prove, back up
   - Examples: "Find evidence for our pacemaker safety claims", "Research competitor analysis for stents"

2. CLAIM_GENERATION: Generating new marketing content, claims, or promotional materials  
   - Keywords: generate, create, write, develop, make, produce, build, craft, compose, draft
   - Examples: "Generate marketing claims for new cardiac device", "Create promotional content for our latest stent"

Respond in JSON format:
{
  "intent": "evidence_discovery" | "claim_generation" | "clarification_needed",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

If the query is unclear or could be either, use "clarification_needed".""")
    
    try:
        response = llm.invoke([
            intent_prompt,
            HumanMessage(content=f"Classify this Boston Scientific marketing query: {user_query}")
        ])
        
        result = json.loads(response.content.strip())
        intent = result.get("intent", "clarification_needed")
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")
        
        print(f"🎯 Intent Classification: {intent} (confidence: {confidence}) - {reasoning}")
        
        if intent == "evidence_discovery":
            return {
                "intent": "evidence_discovery",
                "selected_server": "rag_server"
            }
        elif intent == "claim_generation":
            return {
                "intent": "claim_generation", 
                "selected_server": "summary_server"
            }
        else:
            # Clarification needed
            clarification_message = """I'd be happy to help with your Boston Scientific marketing needs! 

I can assist with:
🔍 **Evidence Discovery** - Find research, data, and evidence to support your marketing claims
📝 **Claim Generation** - Generate new marketing content and promotional materials

Could you clarify whether you need me to:
• Find evidence/data to support existing claims
• Generate new marketing content for a product

What would you like me to help you with?"""
            
            return {
                "intent": "clarification_needed",
                "messages": [AIMessage(content=clarification_message)]
            }
            
    except Exception as e:
        print(f"⚠️ Intent detection error: {e}")
        return {
            "intent": "clarification_needed",
            "messages": [AIMessage(content="I'd be happy to help with your marketing needs! Could you please clarify what you're looking for?")]
        }

# Evidence Discovery Node - Find evidence for claims
def evidence_discovery_node(state: BeekeeperState) -> dict:
    """🔍 Evidence Discovery: Create tool calls for evidence research"""
    print("🔍 Evidence Discovery: Preparing to search for supporting evidence...")
    
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
    ai_message = AIMessage(content="Searching for evidence to support your marketing claims...", tool_calls=[tool_call])
    
    return {"messages": [ai_message]}

# Claim Generation Node - Generate marketing content
def claim_generation_node(state: BeekeeperState) -> dict:
    """📝 Claim Generation: Create tool calls for marketing content generation"""
    print("📝 Claim Generation: Preparing to generate marketing content...")
    
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
    
    # Check if query contains product information for marketing claims
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4"))
    
    product_detection_prompt = SystemMessage(content="""You are analyzing a Boston Scientific marketing query to determine if it contains specific product information for generating marketing claims.

Respond in JSON format:
{
  "has_product": true/false,
  "product_name": "extracted product name" (if found),
  "confidence": 0.0-1.0
}

Look for:
- Explicit product mentions ("claims for pacemaker", "marketing for cardiac stent") 
- Medical device categories ("defibrillator marketing", "catheter claims")
- Boston Scientific product lines ("generate content for our new device")

If no specific product/device is mentioned, set has_product to false.""")
    
    try:
        detection_response = llm.invoke([
            product_detection_prompt,
            HumanMessage(content=f"Analyze this Boston Scientific query for product information: {user_query}")
        ])
        
        detection_result = json.loads(detection_response.content.strip())
        has_product = detection_result.get("has_product", False)
        product_name = detection_result.get("product_name", "")
        confidence = detection_result.get("confidence", 0.0)
        
        print(f"📝 Product Detection: has_product={has_product}, product={product_name}, confidence={confidence}")
        
        if not has_product or confidence < 0.7:
            # Return clarification request
            clarification_message = """I'd be happy to generate marketing claims for you! However, I need to know what specific Boston Scientific product or device you'd like me to focus on.

For example, you could ask:
• "Generate marketing claims for our new pacemaker"
• "Create promotional content for cardiac stents"
• "Develop marketing materials for defibrillator technology"
• "Write claims about our catheter innovations"

Could you please specify which Boston Scientific product or medical device you'd like marketing content for?"""
            
            return {
                "intent": "clarification_needed",
                "messages": [AIMessage(content=clarification_message)]
            }
            
    except Exception as e:
        print(f"⚠️ Product detection error: {e}")
        # On error, ask for clarification
        clarification_message = """I'd be happy to help generate marketing claims! Could you please specify which Boston Scientific product or medical device you'd like me to focus on?

For example: "Generate marketing claims for [product name]" or "Create content for [device type]"."""
        
        return {
            "intent": "clarification_needed", 
            "messages": [AIMessage(content=clarification_message)]
        }
    
    # Create tool call for summarization server
    tool_call = {
        "name": "call_mcp_tool",
        "args": {
            "server_key": "summary_server",
            "tool_name": "generate_medical_device_claims",
            "tool_args": {"product_name": product_name or user_query}
        },
        "id": f"call_{uuid4()}"
    }
    
    # Create AI message with tool call
    ai_message = AIMessage(content="Generating marketing content for your Boston Scientific product...", tool_calls=[tool_call])
    
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

def route_after_chat(state: BeekeeperState) -> str:
    """Route after chat node based on conversation state"""
    conversation_complete = state.get("conversation_complete", False)
    
    print(f"🔄 Routing after chat: complete={conversation_complete}")
    
    # If conversation is complete, end the flow
    if conversation_complete:
        return "END"
    
    # Otherwise, route to intent detection
    return "intent_detection"

def route_after_intent_detection(state: BeekeeperState) -> str:
    """Route after intent detection based on intent"""
    intent = state.get("intent")
    
    print(f"🔄 Routing after intent detection: intent={intent}")
    
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

# Create the  Agent Graph
def create_beekeeper_agent():
    print("Creating Agent with clean workflow...")
    
    # Create graph
    workflow = StateGraph(BeekeeperState)
    
    # Add nodes
    workflow.add_node("chat", chat_node)
    workflow.add_node("intent_detection", intent_detection_node)
    workflow.add_node("evidence_discovery", evidence_discovery_node)
    workflow.add_node("claim_generation", claim_generation_node)
    
    # Add standard LangGraph tool nodes - streaming happens automatically via get_stream_writer()
    workflow.add_node("rag_tool", ToolNode([call_mcp_tool]))
    workflow.add_node("summary_tool", ToolNode([call_mcp_tool]))
    
    # Set entry point
    workflow.set_entry_point("chat")
    
    # Add conditional routing from chat
    workflow.add_conditional_edges(
        "chat",
        route_after_chat,
        {
            "intent_detection": "intent_detection",
            "END": END
        }
    )
    
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
    
    # Add edges from tool nodes back to chat for conversational responses
    workflow.add_edge("rag_tool", "chat")
    workflow.add_edge("summary_tool", "chat")
    
    # Compile the graph
    app = workflow.compile()
    
    print("Boston Scientific Marketing Agent created successfully!")
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
    """Interactive main function for testing Boston Scientific Marketing workflow"""
    print("Starting Boston Scientific Marketing Agent...")
    print("Architecture: Chat → Intent Detection → Evidence Discovery OR Marketing Claims → Tools → Chat Response")
    print("Specialized for Boston Scientific's marketing team workflows")
    
    # Test server connections
    await test_server_connections()
    
    print(f"\nBoston Scientific Marketing Workflow:")
    print(f"• Chat: Handle user conversation and tool result responses")
    print(f"• Intent Detection: Classify evidence_discovery vs claim_generation")
    print(f"• Evidence Discovery: Find evidence to support marketing claims")
    print(f"• Claim Generation: Generate new marketing content for products")
    print(f"• Generic MCP Tool: Single function handles all server/tool combinations")
    print(f"• Professional Flow: Each step tailored for marketing team needs")
    
    print("\nExample queries for Boston Scientific:")
    print("• 'Find evidence for our pacemaker safety claims' → Evidence Discovery → RAG ToolNode → Chat Response")
    print("• 'Generate marketing claims for new cardiac stent' → Claim Generation → Summary ToolNode → Chat Response")
    print("• 'Research competitor analysis for defibrillators' → Evidence Discovery → RAG ToolNode → Chat Response")
    print("• 'Create promotional content for catheter technology' → Claim Generation → Summary ToolNode → Chat Response")
    print("• 'Generate claims' (no product specified) → LLM detects missing product → Asks for clarification")
    print("• 'Help with marketing' → Intent Detection provides clarification with Boston Scientific context")
    print("• Type 'quit' to exit")
    print("-" * 80)
    
    try:
        while True:
            user_input = input(f"\nBoston Scientific Marketing Team: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            if not user_input:
                continue
            
            print(f"\nMarketing Assistant: ", end="", flush=True)
            
            # Process the query through Boston Scientific workflow
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