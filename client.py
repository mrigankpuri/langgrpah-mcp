#!/usr/bin/env python3
"""
LangGraph Agent with official MCP streaming client
"""

import asyncio
import json
import aiohttp
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# State definition for LangGraph
class AgentState(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    streaming_messages: List[str] = Field(default_factory=list)
    last_tool_result: Optional[str] = None


class MCPStreamingClient:
    """Official MCP streaming client"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def list_tools(self) -> Dict[str, Any]:
        """List available tools from MCP server"""
        try:
            async with self.session.post(
                    f"{self.base_url}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {}
                    }
            ) as response:
                result = await response.json()
                return result.get("result", {})
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return {}

    async def call_tool_with_streaming(
            self,
            tool_name: str,
            arguments: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Call MCP tool and stream responses"""

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        streaming_messages = []
        final_result = None

        try:
            async with self.session.post(
                    f"{self.base_url}/mcp",
                    json=payload,
                    headers={
                        'Accept': 'application/json, text/event-stream',
                        'Cache-Control': 'no-cache'
                    }
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    yield {
                        "type": "error",
                        "message": f"HTTP {response.status}: {error_text}",
                        "timestamp": datetime.now().isoformat()
                    }
                    return

                # Handle streaming response
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()

                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])  # Remove 'data: ' prefix

                            # Handle different message types
                            if 'method' in data and data['method'] == 'notifications/message':
                                # This is a streaming log message
                                params = data.get('params', {})
                                log_level = params.get('level', 'info')
                                log_data = params.get('data', '')
                                logger_name = params.get('logger', 'unknown')

                                stream_msg = {
                                    "type": "stream",
                                    "level": log_level,
                                    "message": log_data,
                                    "logger": logger_name,
                                    "timestamp": datetime.now().isoformat()
                                }

                                streaming_messages.append(log_data)
                                yield stream_msg

                            elif 'result' in data:
                                # This is the final result
                                final_result = data['result']
                                yield {
                                    "type": "result",
                                    "content": final_result,
                                    "streaming_messages": streaming_messages,
                                    "timestamp": datetime.now().isoformat()
                                }
                                break

                            elif 'error' in data:
                                # This is an error response
                                error = data['error']
                                yield {
                                    "type": "error",
                                    "message": error.get('message', 'Unknown error'),
                                    "code": error.get('code', -1),
                                    "timestamp": datetime.now().isoformat()
                                }
                                break

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON: {e}, line: {line_str}")
                            continue

                    elif line_str == '':
                        # Empty line, continue
                        continue

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {
                "type": "error",
                "message": f"Connection error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }


# Global MCP client
mcp_client = None


async def initialize_mcp_client():
    """Initialize the global MCP client"""
    global mcp_client
    mcp_client = MCPStreamingClient()
    await mcp_client.__aenter__()


@tool
async def streaming_task_tool(task_name: str, steps: int = 3, delay: float = 1.0) -> str:
    """
    Execute a streaming task with progress updates.

    Args:
        task_name: Name of the task to execute
        steps: Number of steps in the task (1-10)
        delay: Delay between steps in seconds (0.1-5.0)
    """
    if not mcp_client:
        return "Error: MCP client not initialized"

    streaming_messages = []
    final_result = None

    try:
        async for update in mcp_client.call_tool_with_streaming(
                "streaming_task",
                {
                    "task_name": task_name,
                    "steps": steps,
                    "delay": delay
                }
        ):
            logger.info(f"Received update: {update} \n\n")

            if update["type"] == "stream":
                streaming_messages.append(f"📝 {update['message']}")
            elif update["type"] == "result":
                final_result = update["content"]
                streaming_messages.extend([f"📝 {msg}" for msg in update.get("streaming_messages", [])])
            elif update["type"] == "error":
                return f"❌ Error: {update['message']}"

    except Exception as e:
        return f"❌ Tool execution failed: {str(e)}"

    # Format response
    response_parts = ["🚀 Streaming Task Execution:"]
    response_parts.extend(streaming_messages)

    if final_result:
        if isinstance(final_result, list) and len(final_result) > 0:
            content = final_result[0]
            if isinstance(content, dict) and "text" in content:
                response_parts.append(f"✅ Final Result: {content['text']}")
            else:
                response_parts.append(f"✅ Final Result: {str(final_result)}")

    return "\n".join(response_parts)


@tool
async def data_processing_tool(dataset: str = "sample_data", batch_size: int = 100, total_records: int = 1000) -> str:
    """
    Process data with streaming batch updates.

    Args:
        dataset: Name of the dataset to process
        batch_size: Number of records per batch (10-1000)
        total_records: Total number of records to process (1-10000)
    """
    if not mcp_client:
        return "Error: MCP client not initialized"

    streaming_messages = []
    final_result = None

    try:
        async for update in mcp_client.call_tool_with_streaming(
                "data_processing",
                {
                    "dataset": dataset,
                    "batch_size": batch_size,
                    "total_records": total_records
                }
        ):
            logger.info(f"Received update: {update}")

            if update["type"] == "stream":
                streaming_messages.append(f"📊 {update['message']}")
            elif update["type"] == "result":
                final_result = update["content"]
            elif update["type"] == "error":
                return f"❌ Error: {update['message']}"

    except Exception as e:
        return f"❌ Data processing failed: {str(e)}"

    # Format response
    response_parts = ["💾 Data Processing Results:"]
    response_parts.extend(streaming_messages)

    if final_result:
        if isinstance(final_result, list) and len(final_result) > 0:
            content = final_result[0]
            if isinstance(content, dict) and "text" in content:
                response_parts.append(f"✅ Summary: {content['text']}")

    return "\n".join(response_parts)


@tool
async def health_check_tool() -> str:
    """
    Perform system health check with streaming component status.
    """
    if not mcp_client:
        return "Error: MCP client not initialized"

    streaming_messages = []
    final_result = None

    try:
        async for update in mcp_client.call_tool_with_streaming("health_check", {}):
            logger.info(f"Received update: {update}")

            if update["type"] == "stream":
                streaming_messages.append(f"🔍 {update['message']}")
            elif update["type"] == "result":
                final_result = update["content"]
            elif update["type"] == "error":
                return f"❌ Error: {update['message']}"

    except Exception as e:
        return f"❌ Health check failed: {str(e)}"

    # Format response
    response_parts = ["🏥 System Health Check:"]
    response_parts.extend(streaming_messages)

    if final_result:
        if isinstance(final_result, list) and len(final_result) > 0:
            content = final_result[0]
            if isinstance(content, dict) and "text" in content:
                response_parts.append(f"✅ Status: {content['text']}")

    return "\n".join(response_parts)


# Create tool node
tools = [streaming_task_tool, data_processing_tool, health_check_tool]
tool_node = ToolNode(tools)


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end"""
    messages = state.messages
    last_message = messages[-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END


async def call_model(state: AgentState) -> Dict[str, Any]:
    """Model that decides when to use streaming tools"""
    messages = state.messages
    last_message = messages[-1] if messages else None

    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()

        # Determine which tool to use based on user input
        if any(word in content for word in ["task", "stream", "progress"]):
            response = AIMessage(
                content="I'll execute a streaming task for you.",
                tool_calls=[{
                    "name": "streaming_task_tool",
                    "args": {
                        "task_name": "user_requested_task",
                        "steps": 4,
                        "delay": 1.0
                    },
                    "id": "call_1"
                }]
            )
        elif any(word in content for word in ["data", "process", "batch"]):
            response = AIMessage(
                content="I'll start data processing with streaming updates.",
                tool_calls=[{
                    "name": "data_processing_tool",
                    "args": {
                        "dataset": "user_dataset",
                        "batch_size": 150,
                        "total_records": 800
                    },
                    "id": "call_2"
                }]
            )
        elif any(word in content for word in ["health", "check", "status"]):
            response = AIMessage(
                content="I'll perform a system health check.",
                tool_calls=[{
                    "name": "health_check_tool",
                    "args": {},
                    "id": "call_3"
                }]
            )
        else:
            response = AIMessage(
                content="""Hello! I can help you with streaming operations:

• Say "run a streaming task" - Execute a task with progress updates
• Say "process data" - Run data processing with batch updates  
• Say "health check" - Check system status with component details

All operations provide real-time streaming updates!"""
            )
    elif isinstance(last_message, ToolMessage):
        response = AIMessage(
            content=f"Here are the streaming results:\n\n{last_message.content}"
        )
    else:
        response = AIMessage(
            content="I'm ready to help with streaming MCP operations!"
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
    print("🚀 Starting LangGraph Agent with Official MCP Streaming...")

    # Initialize MCP client
    await initialize_mcp_client()

    try:
        # Test connection
        tools_info = await mcp_client.list_tools()
        print(f"✅ Connected to MCP server! Available tools: {len(tools_info.get('tools', []))}")

        print("\n💡 Try these commands:")
        print("• 'run a streaming task' - Execute task with progress")
        print("• 'process some data' - Run data processing")
        print("• 'health check' - Check system status")
        print("• 'quit' - Exit")
        print("-" * 50)

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
            async for output in app.astream(initial_state):
                for key, value in output.items():
                    if key == "agent" and "messages" in value:
                        last_message = value["messages"][-1]
                        if isinstance(last_message, AIMessage):
                            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                                print(last_message.content)
                    elif key == "tools" and "messages" in value:
                        # Tool execution completed - results are shown via agent
                        pass

    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Clean up MCP client
        if mcp_client:
            await mcp_client.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())