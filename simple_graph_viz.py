#!/usr/bin/env python
"""
🐝 Beekeeper Agent - Simple Graph Visualization
"""
from beekeeper_agent import create_beekeeper_agent
from IPython.display import Image, display

def main():
    """Generate and display the Beekeeper graph"""
    print("🐝 Generating Beekeeper Agent Graph...")
    
    # Create the graph
    graph = create_beekeeper_agent()
    
    # Try to display the graph
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
        print("✅ Graph displayed successfully!")
    except Exception as e:
        print(f"⚠️ Display failed: {e}")
        print("💡 This requires some extra dependencies and is optional")
    
    # Also save to file
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("beekeeper_graph.png", "wb") as f:
            f.write(png_data)
        print("✅ Graph saved as: beekeeper_graph.png")
    except Exception as e:
        print(f"⚠️ Save failed: {e}")

if __name__ == "__main__":
    main() 