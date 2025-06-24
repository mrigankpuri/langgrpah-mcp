#!/usr/bin/env python
"""
🐝 Beekeeper FastAPI - Streaming REST API for Beekeeper Agent with Intent Detection
"""
import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator
import uvicorn
import logging

# Import the Beekeeper agent with intent detection business flow
from beekeeper_agent import process_query, test_server_connections

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="🐝 Beekeeper Agent - Streaming API",
    description="Streaming REST API for Beekeeper Agent with Intent Detection Business Flow",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    query: str

async def stream_beekeeper_response(query: str) -> AsyncGenerator[str, None]:
    """Real-time streaming response using LangGraph's custom streaming mode"""
    try:
        # Create the Beekeeper agent
        from beekeeper_agent import create_beekeeper_agent
        from langchain_core.messages import HumanMessage, ToolMessage
        
        app_agent = create_beekeeper_agent()
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "intent": None,
            "selected_server": None,
            "conversation_complete": False
        }
        
        # Stream using both 'updates' and 'custom' modes to get everything
        async for mode, chunk in app_agent.astream(initial_state, stream_mode=["updates", "custom"]):
            
            if mode == "custom":
                # Real-time MCP notifications from get_stream_writer()
                if isinstance(chunk, dict):
                    if "mcp_notification" in chunk:
                        notification = chunk["mcp_notification"]
                        notification_type = chunk.get("type", "unknown")
                        # Only show progress updates, not technical details
                        if notification_type == "mcp_progress":
                            yield f"data: {notification}\n\n"
            
            elif mode == "updates":
                # Standard node execution updates
                node_name = list(chunk.keys())[0]
                node_state = chunk[node_name]
                
                # Show node name
                yield f"data: **{node_name.replace('_', ' ').title()}**\n\n"
                
                # Stream messages as they come
                messages = node_state.get('messages', [])
                if messages:
                    last_message = messages[-1]
                    
                    # Show tool calls
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            tool_args = tool_call.get('args', {})
                            yield f"data: **Calling tool:** {tool_name}\n\n"
                            yield f"data: **Arguments:** {tool_args}\n\n"
                    
                    # Show message content
                    if hasattr(last_message, 'content') and last_message.content.strip():
                        yield f"data: {last_message.content}\n\n"
                    
                    # Show tool results
                    if isinstance(last_message, ToolMessage):
                        yield f"data: **Tool Result:**\n\n"
                        content = last_message.content
                        # Send content as-is since UI now handles large chunks properly
                        yield f"data: {content}\n\n"
        
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"
    
    finally:
        yield "data: [DONE]\n\n"

@app.get("/")
async def root():
    """Root endpoint with Beekeeper info"""
    return {
        "service": "🐝 Beekeeper Agent - Streaming API",
        "version": "2.0.0",
        "business_flow": "Intent Detection → Evidence Discovery OR Claim Generation → Tools",
        "endpoints": {
            "chat": "/chat/stream",
            "health": "/health"
        }
    }

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Beekeeper streaming chat endpoint with Server-Sent Events and Intent Detection"""
    try:
        logger.info(f"Processing Beekeeper streaming chat request: {request.query}")
        
        return StreamingResponse(
            stream_beekeeper_response(request.query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check for Beekeeper agent and MCP servers"""
    try:
        # Test both MCP servers using the beekeeper_agent function
        print("🔍 Health check: Testing MCP server connections...")
        
        # Import here to avoid circular imports
        from beekeeper_agent import get_mcp_client, server_config
        
        server_status = {}
        
        # Test rag_server (RAG)
        try:
            async with get_mcp_client("rag_server") as (client1, server1_info):
                if client1 and server1_info:
                    await client1.list_tools()
                    server_status["rag_server"] = {
                        "status": "connected",
                        "name": server1_info["name"],
                        "purpose": server1_info["purpose"]
                    }
                else:
                    server_status["rag_server"] = {"status": "disconnected", "name": "RAG Server"}
        except Exception as e:
            server_status["rag_server"] = {"status": f"error: {str(e)[:50]}", "name": "RAG Server"}
        
        # Test summary_server (Summarization)
        try:
            async with get_mcp_client("summary_server") as (client2, server2_info):
                if client2 and server2_info:
                    await client2.list_tools()
                    server_status["summary_server"] = {
                        "status": "connected",
                        "name": server2_info["name"],
                        "purpose": server2_info["purpose"]
                    }
                else:
                    server_status["summary_server"] = {"status": "disconnected", "name": "Summarization Server"}
        except Exception as e:
            server_status["summary_server"] = {"status": f"error: {str(e)[:50]}", "name": "Summarization Server"}
        
        return {
            "service": "🐝 Beekeeper Agent",
            "api_status": "healthy",
            "business_flow": "Intent Detection → Evidence Discovery OR Claim Generation → Tools",
            "mcp_servers": server_status,
            "endpoints": {
                "chat_stream": "/chat/stream",
                "health": "/health"
            }
        }
        
    except Exception as e:
        return {
            "service": "🐝 Beekeeper Agent",
            "api_status": f"error: {str(e)}",
            "business_flow": "Intent Detection → Evidence Discovery OR Claim Generation → Tools",
            "mcp_servers": "unknown",
            "endpoints": {
                "chat_stream": "/chat/stream",
                "health": "/health"
            }
        }

if __name__ == "__main__":
    print("🐝 Starting Beekeeper FastAPI Server...")
    print("🏗️  Business Flow: Intent Detection → Evidence Discovery OR Claim Generation → Tools")
    print("🌐 Server will be available at: http://localhost:8003")
    print("📡 Streaming endpoint: http://localhost:8003/chat/stream")
    print("🔍 Health check: http://localhost:8003/health")
    
    uvicorn.run(
        "beekeeper_fastapi:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    ) 