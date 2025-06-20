# Universal FastMCP Streaming Client

A **generic streaming MCP client** that can connect to **any FastMCP server** with real-time streaming capabilities. Built with the **proper LangGraph pattern** using `bind_tools()` and `tools_condition` for intelligent LLM-driven tool selection.

## 🚀 Features

- **Universal FastMCP Client** - Connect to any FastMCP server by URL
- **Smart LLM Tool Selection** - Uses proper LangGraph `bind_tools()` pattern 
- **Real-time streaming** progress updates during tool execution  
- **Natural Language Interface** - Ask questions and request tools naturally
- **Configuration management** for multiple MCP servers
- **Auto-discovery** of available tools from any server

## 📋 Files Overview

### Core Files
- `generic_streaming_client.py` - **Universal streaming client** with proper LangGraph pattern ⭐
- `mcp_launcher.py` - **Configuration-based launcher** for managing multiple servers
- `server.py` - Example FastMCP weather server (for testing)

### Legacy Files (for reference)
- `streaming_client.py` - Original weather-specific client  
- `client.py` - Simple client using MultiServerMCPClient (no streaming)

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API
Create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Connect to Any FastMCP Server

#### Option A: Direct Connection
```bash
python generic_streaming_client.py --url http://localhost:8001/mcp --name "My Server"
```

#### Option B: Using Configuration Manager
```bash
# List available servers
python mcp_launcher.py list

# Add a new server
python mcp_launcher.py add

# Connect to configured server
python mcp_launcher.py connect weather
```

## 🌟 Smart Tool Selection

### LangGraph Pattern with bind_tools()
The client uses the **proper LangGraph pattern** for intelligent tool calling:

```python
# LLM decides which tools to call using bind_tools()
llm_with_tools = llm.bind_tools(tools)
response = llm_with_tools.invoke(messages)

# Graph uses tools_condition for routing
graph_builder.add_conditional_edges("chatbot", tools_condition)
```

### Natural Language Tool Calling
Just ask naturally - the LLM understands and calls the right tools:

```
🧑 You: What's the weather in Tokyo?
🤖 Agent: [Automatically calls get_weather tool]

📡 🔧 Step 1/3: Executing get_weather...
📡 🔧 Step 2/3: Executing get_weather...
📡 🔧 Step 3/3: Executing get_weather...

The weather in Tokyo is always sunny!

🧑 You: What tools do you have?
🤖 Agent: I have access to the Weather API tool, which allows me to retrieve weather information for a specific city...
```

### Connect to Any Server
```bash
# Connect to weather server
python generic_streaming_client.py --url http://localhost:8001/mcp --name "Weather API"

# Connect to file system server  
python generic_streaming_client.py --url http://localhost:8002/mcp --name "File System"

# Connect to database server
python generic_streaming_client.py --url http://localhost:8003/mcp --name "Database API"
```

### Auto-Discovery of Tools
The client automatically discovers and lists all available tools from any FastMCP server:

```
✅ Connected to MCP server!
📋 Available tools: ['get_weather', 'get_forecast', 'search_location']

🔧 Tool descriptions:
  • get_weather: Get current weather for a city
  • get_forecast: Get weather forecast for multiple days
  • search_location: Find location coordinates
```

## 🔧 Configuration Management

### Create Server Configuration
```bash
python mcp_launcher.py add
```

This creates/updates `mcp_servers.json`:
```json
{
  "servers": {
    "weather": {
      "url": "http://localhost:8001/mcp",
      "name": "Weather Server", 
      "description": "FastMCP weather server with streaming",
      "model": "gpt-3.5-turbo"
    },
    "filesystem": {
      "url": "http://localhost:8002/mcp",
      "name": "File System Server",
      "description": "File operations with streaming", 
      "model": "gpt-4"
    }
  }
}
```

### Connect to Configured Servers
```bash
# List all configured servers
python mcp_launcher.py list

# Connect to specific server
python mcp_launcher.py connect weather

# Connect with different model
python mcp_launcher.py connect weather --model gpt-4
```

## 🎯 Usage Examples

### Intelligent Conversation
The LLM automatically understands user intent and calls appropriate tools:

```
🧑 You: Get weather for Paris
🤖 Agent: [Calls get_weather automatically with city="Paris"]

🧑 You: What can you do?
🤖 Agent: I can help you get weather information using the get_weather tool...

🧑 You: How's the weather in London today?
🤖 Agent: [Calls get_weather automatically with city="London"]
```

### Smart Error Handling
```
🧑 You: Get me data for invalid_tool
🤖 Agent: Error: Tool 'invalid_tool' not available. Available tools: get_weather
```

## 🔍 Architecture

```
User Input → LangGraph Agent → LLM with bind_tools() → Generic MCP Tool → Any FastMCP Server → Streaming Updates → Client Display
```

### Key Components:
- **Smart LLM Router**: Uses `bind_tools()` for intelligent tool selection
- **Generic Tool Caller**: `call_mcp_tool_streaming()` - works with any FastMCP tool
- **Auto-Discovery**: Automatically finds available tools from any server
- **Streaming Handler**: Real-time progress updates with `📡` indicators
- **Configuration Manager**: Save and manage multiple server connections

## 🌐 Supported Server Types

This client works with **any FastMCP server** that supports:
- HTTP streaming transport (`http://localhost:port/mcp`)
- Progress notifications (`ctx.report_progress()`)
- Standard MCP tool calling

### Tested With:
- ✅ Weather servers
- ✅ File system servers  
- ✅ Database servers
- ✅ API integration servers
- ✅ Custom FastMCP implementations

## 🛡️ Error Handling

The system gracefully handles:
- **Server connection failures**: Clear error messages with troubleshooting tips
- **Invalid tool names**: Lists available tools when tool not found
- **LLM-driven validation**: Smart error recovery through conversation
- **Streaming interruptions**: Automatic cleanup and reconnection
- **Unknown servers**: Configuration management for easy setup

## 📝 Requirements

- Python 3.10+
- OpenAI API key
- FastMCP 2.8.1+
- LangGraph 0.4.8+

## 🎉 Universal Success!

You now have a **universal FastMCP streaming client** with **intelligent LLM-driven tool selection** that can:
- ✅ Connect to **any FastMCP server** by URL
- ✅ **Smart tool calling** using proper LangGraph `bind_tools()` pattern
- ✅ **Natural language interface** - just ask what you want!
- ✅ Auto-discover available tools and capabilities
- ✅ Stream real-time progress from any tool execution  
- ✅ Manage multiple server configurations

**The `📡` emoji indicators show real-time streaming from any FastMCP server with intelligent LLM tool selection!**

## 🚀 Getting Started with Your Own Server

1. **Start your FastMCP server** (any server with HTTP streaming)
2. **Connect directly**: `python generic_streaming_client.py --url YOUR_SERVER_URL`
3. **Or configure it**: `python mcp_launcher.py add` 
4. **Start chatting**: Ask naturally - the LLM will call the right tools automatically!

Transform any FastMCP server into an intelligent streaming-enabled agent with **zero configuration**! 🌟 