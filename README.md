# 🐝 Beekeeper Agent - MCP Streaming Implementation

Real-time streaming agent that automatically detects if you need **evidence discovery** or **content generation**.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Services (4 Terminals)
```bash
# Terminal 1: RAG Server
python mcp_rag_server.py

# Terminal 2: Summary Server  
python mcp_summary_server.py

# Terminal 3: Beekeeper API
python beekeeper_fastapi.py

# Terminal 4: UI
streamlit run simple_beekeeper_ui.py --server.port 8501
```

### 3. Open UI
**URL**: http://localhost:8501

## 💡 How It Works

**Evidence Discovery** (searches for information):
- "Find evidence about ymesh"
- "Research API performance" 
- "Search for security vulnerabilities"

**Content Generation** (creates summaries):
- "Generate summary for iPhone 15"
- "Create report about Tesla earnings"
- "Write overview of AI trends"

## 📡 Real-time Streaming

You'll see live progress updates:
```
Searching for evidence...
Step 1/5: Analyzing query for semantic search
Step 2/5: Searching document embeddings database  
Step 3/5: Ranking and filtering relevant documents
Step 4/5: Extracting key passages and evidence
Step 5/5: Compiling comprehensive evidence summary

Based on my search, here's what I found...
```

## 🔧 Architecture

- **Intent Detection**: LLM classifies your query automatically
- **Evidence Discovery**: Routes to RAG Server (Port 8001) 
- **Content Generation**: Routes to Summary Server (Port 8002)
- **Streaming**: Real-time progress updates via FastAPI (Port 8003)

## 🧪 Testing

```bash
# Test connections
python test_connection.py

# Test agent directly
python beekeeper_agent.py
```

## 📁 Core Files

- `beekeeper_agent.py` - Main LangGraph agent
- `beekeeper_fastapi.py` - Streaming API server
- `simple_beekeeper_ui.py` - Streamlit UI
- `mcp_rag_server.py` - Evidence discovery server
- `mcp_summary_server.py` - Content generation server

That's it! The agent automatically figures out what you need and streams results in real-time. 🐝 