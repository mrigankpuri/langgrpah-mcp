#!/usr/bin/env python
"""
Test script to verify MCP server connection and basic functionality
"""
import asyncio
from fastmcp import Client
import sys

async def test_connection():
    """Test connection to the Beekeeper MCP server"""
    print("🔍 Testing connection to Beekeeper MCP Server...")
    print("=" * 50)
    
    try:
        client = Client("http://localhost:8002/mcp")
        await client.__aenter__()
        
        print("✅ Connected to MCP server successfully!")
        
        # Test listing tools
        print("\n📋 Available tools:")
        try:
            tools_response = await client.list_tools()
            # Handle different possible response formats
            if hasattr(tools_response, 'tools'):
                tools = tools_response.tools
            elif isinstance(tools_response, list):
                tools = tools_response
            else:
                tools = []
                
            for tool in tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    print(f"  - {tool.name}: {tool.description}")
                else:
                    print(f"  - {tool}")
                    
            if not tools:
                print("  No tools found or unable to parse tools list")
                
        except Exception as e:
            print(f"  ⚠️ Could not list tools: {e}")
        
        # Test a simple tool call
        print("\n🔧 Testing RAG tool...")
        def progress_handler(progress_token, progress, total, message=None):
            print(f"📡 Progress: {progress}/{total if total else '?'}")
        
        try:
            result = await client.call_tool(
                "rag_tool",
                {"query": "test connection"},
                progress_handler=progress_handler
            )
            
            print(f"✅ RAG tool result: {result[0].text if result else 'No result'}")
        except Exception as e:
            print(f"⚠️ RAG tool test failed: {e}")
        
        await client.__aexit__(None, None, None)
        print("\n🎉 Connection test completed!")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Make sure the Beekeeper server is running:")
        print("   python beekeeper_server.py")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_connection()) 