#!/usr/bin/env python
"""
🐝 Test Beekeeper Agent - Intent Detection Business Flow
"""
import asyncio
from beekeeper_agent import process_query, test_server_connections

async def test_intent_detection():
    """Test LLM-based intent detection with various queries including clarification cases"""
    print("🎯 Testing LLM-Based Intent Detection with Clarification...")
    print("=" * 60)
    
    # Test queries for each intent with more diverse examples
    test_queries = [
        # Evidence Discovery queries
        ("Find evidence about API performance", "evidence_discovery"),
        ("Search for database optimization data", "evidence_discovery"),
        ("What are the security vulnerabilities in our system?", "evidence_discovery"),
        ("Research user behavior patterns from last month", "evidence_discovery"),
        ("Investigate the root cause of system bottlenecks", "evidence_discovery"),
        ("Analyze the performance metrics for Q3", "evidence_discovery"),
        ("Show me data about customer churn rates", "evidence_discovery"),
        ("Look up information on competitor analysis", "evidence_discovery"),
        
        # Claim Generation queries
        ("Create a summary for analytics platform", "claim_generation"),
        ("Generate a report on user engagement", "claim_generation"),
        ("Write a product overview for our new service", "claim_generation"),
        ("Produce a technical specification document", "claim_generation"),
        ("Build a market analysis report", "claim_generation"),
        ("Compose a quarterly business review", "claim_generation"),
        ("Develop a proposal for the new initiative", "claim_generation"),
        ("Make a summary of the meeting outcomes", "claim_generation"),
        
        # Clarification needed queries
        ("Help me with the project", "clarification_needed"),
        ("What about performance?", "clarification_needed"),
        ("I need something about analytics", "clarification_needed"),
        ("Can you help me?", "clarification_needed"),
        ("Tell me about the system", "clarification_needed"),
        ("I have a question", "clarification_needed"),
        ("Find and create a report about performance", "clarification_needed"),  # mixed intent
        ("Research data and then generate a summary", "clarification_needed"),  # mixed intent
    ]
    
    correct_predictions = 0
    total_predictions = len(test_queries)
    
    for query, expected_intent in test_queries:
        print(f"\n📝 Query: '{query}'")
        print(f"🎯 Expected Intent: {expected_intent}")
        
        try:
            result = await process_query(query)
            actual_intent = result.get("intent")
            selected_server = result.get("selected_server")
            
            print(f"🤖 LLM Classified: {actual_intent}")
            
            if actual_intent == expected_intent:
                print(f"✅ Intent Detection: PASS")
                correct_predictions += 1
                
                # Check server routing consistency
                if expected_intent == "evidence_discovery" and selected_server == "rag_server":
                    print(f"✅ Server Routing: PASS ({selected_server})")
                elif expected_intent == "claim_generation" and selected_server == "summary_server":
                    print(f"✅ Server Routing: PASS ({selected_server})")
                elif expected_intent == "clarification_needed" and selected_server is None:
                    print(f"✅ Server Routing: PASS (no server selected for clarification)")
                else:
                    print(f"❌ Server Routing: INCONSISTENT")
                    
            else:
                print(f"❌ Intent Detection: FAIL (got {actual_intent}, expected {expected_intent})")
                
            # Check if clarification response is provided
            if actual_intent == "clarification_needed":
                messages = result.get("messages", [])
                if messages and hasattr(messages[-1], 'content'):
                    response_content = messages[-1].content
                    if "clarification" in response_content.lower() and "could you please" in response_content.lower():
                        print(f"✅ Clarification Response: PROVIDED")
                    else:
                        print(f"⚠️ Clarification Response: INCOMPLETE")
                        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)
    
    # Print summary
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n📊 Intent Detection Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
    
    if accuracy >= 80:
        print("✅ Intent detection performance: GOOD")
    elif accuracy >= 60:
        print("⚠️ Intent detection performance: ACCEPTABLE")
    else:
        print("❌ Intent detection performance: NEEDS IMPROVEMENT")

async def test_full_business_flow():
    """Test the complete business flow with real MCP servers"""
    print("\n🏗️ Testing Full Business Flow...")
    print("=" * 60)
    
    # Test Evidence Discovery flow
    print("\n🔍 Testing Evidence Discovery Flow:")
    evidence_query = "Find evidence about API performance metrics"
    print(f"Query: '{evidence_query}'")
    
    try:
        result = await process_query(evidence_query)
        messages = result.get("messages", [])
        
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                print(f"✅ Evidence Discovery: SUCCESS")
                print(f"📊 Intent: {result.get('intent')}")
                print(f"🔧 Server: {result.get('selected_server')}")
                print(f"📄 Response Length: {len(last_message.content)} characters")
            else:
                print("❌ Evidence Discovery: No response content")
        else:
            print("❌ Evidence Discovery: No messages returned")
            
    except Exception as e:
        print(f"❌ Evidence Discovery Error: {str(e)}")
    
    print("-" * 40)
    
    # Test Claim Generation flow
    print("\n📝 Testing Claim Generation Flow:")
    claim_query = "Create a summary for analytics platform"
    print(f"Query: '{claim_query}'")
    
    try:
        result = await process_query(claim_query)
        messages = result.get("messages", [])
        
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                print(f"✅ Claim Generation: SUCCESS")
                print(f"📊 Intent: {result.get('intent')}")
                print(f"🔧 Server: {result.get('selected_server')}")
                print(f"📄 Response Length: {len(last_message.content)} characters")
            else:
                print("❌ Claim Generation: No response content")
        else:
            print("❌ Claim Generation: No messages returned")
            
    except Exception as e:
        print(f"❌ Claim Generation Error: {str(e)}")

async def main():
    """Main test function"""
    print("🐝 Beekeeper Agent - Business Flow Test")
    print("=" * 60)
    
    # Test server connections first
    print("🔍 Testing MCP Server Connections...")
    await test_server_connections()
    
    # Test intent detection
    await test_intent_detection()
    
    # Test full business flow (only if servers are available)
    print("\n⚠️  Full business flow test requires both MCP servers to be running:")
    print("   Terminal 1: python mcp_rag_server.py")
    print("   Terminal 2: python mcp_summary_server.py")
    print("   (Each server runs as a separate FastAPI app)")
    
    user_input = input("\nRun full business flow test? (y/n): ").strip().lower()
    if user_input == 'y':
        await test_full_business_flow()
    
    print("\n🎉 Beekeeper Agent Test Complete!")
    print("🐝 Business Flow: Intent Detection → Evidence Discovery OR Claim Generation → Tools")

if __name__ == "__main__":
    asyncio.run(main()) 