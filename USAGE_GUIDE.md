# Universal FastMCP Streaming Client - Usage Guide

This guide shows you how to connect to **any FastMCP server** with streaming capabilities.

## 🚀 Quick Start

### Method 1: Direct Connection
```bash
# Connect directly to any FastMCP server
python generic_streaming_client.py --url http://your-server:port/mcp --name "Your Server"

# Examples:
python generic_streaming_client.py --url http://localhost:8001/mcp --name "Weather API"
python generic_streaming_client.py --url http://localhost:8002/mcp --name "File System"
python generic_streaming_client.py --url http://localhost:8003/mcp --name "Database API"
```

### Method 2: Configuration-Based (Recommended)
```bash
# 1. List current servers
python mcp_launcher.py list

# 2. Add your server
python mcp_launcher.py add

# 3. Connect to saved server
python mcp_launcher.py connect your_server_id
```

## 📋 Configuration Management

### Adding Servers
```bash
python mcp_launcher.py add
```

Follow the prompts:
```
➕ Add New MCP Server
------------------------------
Server ID (e.g., 'myserver'): filesys
Server URL (e.g., 'http://localhost:8001/mcp'): http://localhost:8002/mcp
Display Name: File System Server
Description: File operations with streaming
OpenAI Model (default: gpt-3.5-turbo): gpt-4
✅ Added server 'filesys' to configuration!
```

### Listing Servers
```bash
python mcp_launcher.py list
```

Output:
```
🌐 Available MCP Servers:
--------------------------------------------------
📛 weather
   URL: http://localhost:8001/mcp
   Name: Weather Server
   Description: FastMCP weather server with streaming
   Model: gpt-3.5-turbo

📛 filesys
   URL: http://localhost:8002/mcp
   Name: File System Server
   Description: File operations with streaming
   Model: gpt-4
```

### Connecting to Servers
```bash
# Connect to configured server
python mcp_launcher.py connect weather

# Connect with different model
python mcp_launcher.py connect weather --model gpt-4

# Direct connection
python mcp_launcher.py connect --url http://localhost:8001/mcp --name "Direct Weather"
```

## 🔧 Using Tools

### Auto-Discovery
When you connect, the client automatically discovers all available tools:

```
✅ Connected to MCP server!
📋 Available tools: ['get_weather', 'get_forecast', 'search_location']

🔧 Tool descriptions:
  • get_weather: Get current weather for a city
  • get_forecast: Get weather forecast for multiple days
  • search_location: Find location coordinates
```

### Natural Language Tool Calling
```
🧑 You: Get weather for Tokyo
🤖 Agent: I'll call the get_weather tool on Weather Server with streaming updates.

📡 🔧 Step 1/3: Executing get_weather...
📡 🔧 Step 2/3: Executing get_weather...
📡 🔧 Step 3/3: Executing get_weather...

Here's the result from Weather Server:
🔧 Tool Execution Progress for get_weather:
🔧 Step 1/3: Executing get_weather...
🔧 Step 2/3: Executing get_weather...
🔧 Step 3/3: Executing get_weather...
✅ Final Result: It's always sunny in Tokyo.
```

### Structured Tool Calling
For precise control, use JSON arguments:

```
🧑 You: Use get_weather with {"city": "Tokyo", "units": "metric"}
🧑 You: Use search_files with {"pattern": "*.py", "directory": "/src", "recursive": true}
🧑 You: Use query_database with {"sql": "SELECT * FROM users LIMIT 10"}
```

### General Questions
```
🧑 You: What tools are available?
🤖 Agent: Available tools on Weather Server: get_weather, get_forecast, search_location

🧑 You: How do I use the get_forecast tool?
🤖 Agent: The get_forecast tool gets weather forecast for multiple days. You can use it like:
"Use get_forecast with {"city": "Tokyo", "days": 5}"
```

## 🌐 Server Examples

### Weather Server
```bash
python generic_streaming_client.py --url http://localhost:8001/mcp --name "Weather API"
```

Tools: `get_weather`, `get_forecast`
```
Use get_weather with {"city": "Tokyo"}
Use get_forecast with {"city": "Paris", "days": 5}
```

### File System Server
```bash
python generic_streaming_client.py --url http://localhost:8002/mcp --name "File System"
```

Tools: `read_file`, `write_file`, `list_directory`, `search_files`
```
Use read_file with {"path": "/home/user/document.txt"}
Use search_files with {"pattern": "*.py", "directory": "/src"}
Use list_directory with {"path": "/home/user"}
```

### Database Server
```bash
python generic_streaming_client.py --url http://localhost:8003/mcp --name "Database API"
```

Tools: `query`, `execute`, `get_schema`
```
Use query with {"sql": "SELECT * FROM users LIMIT 10"}
Use get_schema with {"table": "users"}
Use execute with {"sql": "UPDATE users SET active = true WHERE id = 1"}
```

## 🛡️ Error Handling

### Server Not Running
```
❌ Failed to connect to MCP server: All connection attempts failed
💡 Make sure the server is running and the URL is correct
```

**Solution**: Start your FastMCP server first

### Tool Not Found
```
Error: Tool 'invalid_tool' not available. Available tools: get_weather, get_forecast
```

**Solution**: Use one of the listed available tools

### Invalid JSON Arguments
```
Error: Invalid JSON arguments: Expecting property name enclosed in double quotes
```

**Solution**: Use proper JSON format: `{"key": "value"}`

### OpenAI API Issues
```
Error connecting to LLM: Invalid API key provided
```

**Solution**: Check your `.env` file has correct `OPENAI_API_KEY`

## 📝 Configuration File Format

The `mcp_servers.json` file stores your server configurations:

```json
{
  "servers": {
    "server_id": {
      "url": "http://localhost:8001/mcp",
      "name": "Display Name",
      "description": "Server description",
      "model": "gpt-3.5-turbo"
    }
  }
}
```

You can edit this file directly or use `python mcp_launcher.py add/remove`.

## 🎯 Best Practices

### 1. Use Descriptive Server Names
```json
{
  "weather_prod": {
    "name": "Production Weather API",
    "url": "https://api.weather.com/mcp"
  },
  "weather_dev": {
    "name": "Development Weather API", 
    "url": "http://localhost:8001/mcp"
  }
}
```

### 2. Choose Appropriate Models
- Use `gpt-3.5-turbo` for simple tool calling
- Use `gpt-4` for complex reasoning with multiple tools
- Use `gpt-4-turbo` for large context and advanced reasoning

### 3. Provide Clear JSON Arguments
```bash
# Good
Use get_weather with {"city": "Tokyo", "units": "metric"}

# Bad  
Use get_weather with {city: Tokyo}  # Invalid JSON
```

### 4. Test Connection First
```bash
# Test basic connection
python generic_streaming_client.py --url http://localhost:8001/mcp --name "Test"

# Then add to config if working
python mcp_launcher.py add
```

## 🔧 Advanced Usage

### Custom Models
```bash
python mcp_launcher.py connect weather --model gpt-4
python generic_streaming_client.py --url http://localhost:8001/mcp --model gpt-4-turbo
```

### Server Health Check
The client will show if tools are available and provide descriptions:
```
✅ Connected to MCP server!
📋 Available tools: ['tool1', 'tool2']

🔧 Tool descriptions:
  • tool1: Description of what tool1 does
  • tool2: Description of what tool2 does
```

### Batch Operations
You can pipe commands for automated usage:
```bash
echo -e "Use tool1 with {\"arg\": \"value\"}\nUse tool2 with {\"arg\": \"value\"}\nquit" | python mcp_launcher.py connect server_id
```

## 🎉 You're Ready!

You now have a universal FastMCP streaming client that can:
- ✅ Connect to any FastMCP server
- ✅ Auto-discover available tools  
- ✅ Stream real-time progress updates
- ✅ Handle natural language and structured commands
- ✅ Manage multiple server configurations

**Start connecting to your FastMCP servers and enjoy real-time streaming! 📡** 