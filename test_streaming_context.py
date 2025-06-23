#!/usr/bin/env python
"""
Test streaming via context with MCP server
Demonstrates how to use streaming progress updates through context
"""
import asyncio
from fastmcp import Client

async def test_streaming_context():
    """Test streaming functionality via context"""
    print("🧪 Testing MCP Streaming via Context...")
    print("🌐 Connecting to: http://localhost:8002/mcp")
    
    client = Client("http://localhost:8002/mcp")
    
    try:
        async with client:
            # List available tools
            tools = await client.list_tools()
            print(f"📋 Available tools: {[tool.name for tool in tools]}")
            
            # Test 1: Summarization tool with streaming
            print("\n🔧 Test 1: Summarization with streaming context")
            print("-" * 50)
            
            streaming_messages = []
            
            def progress_handler(progress_token, progress, total, message=None):
                """Handle streaming progress updates"""
                if message:
                    streaming_messages.append(f"📡 {message}")
                    print(f"📡 Streaming: {message}")
                else:
                    progress_msg = f"Progress: {progress}/{total}" if total else f"Progress: {progress}"
                    streaming_messages.append(f"📡 {progress_msg}")
                    print(f"📡 {progress_msg}")
            
            # Call summarization tool with streaming
            result = await client.call_tool(
                "summarization_tool",
                {
                    "product": "StreamingMesh",
                    "output_format": "markdown"
                },
                progress_handler=progress_handler
            )
            
            print(f"\n✅ Final result received:")
            if result and len(result) > 0:
                print(result[0].text[:200] + "..." if len(result[0].text) > 200 else result[0].text)
            
            print(f"\n📊 Streaming messages received: {len(streaming_messages)}")
            
            # Test 2: Weather tool with streaming
            print("\n🔧 Test 2: Weather tool with streaming context")
            print("-" * 50)
            
            streaming_messages_2 = []
            
            def weather_progress_handler(progress_token, progress, total, message=None):
                """Handle weather streaming progress updates"""
                if message:
                    streaming_messages_2.append(f"🌤️ {message}")
                    print(f"🌤️ Streaming: {message}")
            
            # Call weather tool with streaming
            weather_result = await client.call_tool(
                "get_weather",
                {"city": "Tokyo"},
                progress_handler=weather_progress_handler
            )
            
            print(f"\n✅ Weather result:")
            if weather_result and len(weather_result) > 0:
                print(weather_result[0].text)
            
            print(f"\n📊 Weather streaming messages: {len(streaming_messages_2)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_without_streaming():
    """Test the same tools without streaming to compare"""
    print("\n🧪 Testing WITHOUT streaming context (for comparison)...")
    
    client = Client("http://localhost:8002/mcp")
    
    try:
        async with client:
            # Call without progress handler
            result = await client.call_tool(
                "summarization_tool",
                {
                    "product": "NonStreamingMesh", 
                    "output_format": "json"
                }
            )
            
            print("✅ Non-streaming result received")
            if result and len(result) > 0:
                print("📄 Result preview:", result[0].text[:100] + "...")
                
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    """Main test function"""
    print("🚀 MCP Streaming Context Test Suite")
    print("=" * 60)
    
    # Test with streaming
    await test_streaming_context()
    
    # Test without streaming
    await test_without_streaming()
    
    print("\n🎉 Test suite completed!")
    print("\n💡 Key Points:")
    print("• Streaming via context allows real-time progress updates")
    print("• Progress handlers capture streaming messages from the server")
    print("• Context parameter enables server-side streaming functionality")
    print("• Tools work with or without streaming context")

if __name__ == "__main__":
    asyncio.run(main()) 