#!/usr/bin/env python
"""
🐝 Beekeeper Agent - Simple Chat UI
"""
import streamlit as st
import requests
import json

# Configuration
FASTAPI_URL = "http://localhost:8003"

# Page configuration
st.set_page_config(
    page_title="Beekeeper Agent",
    page_icon="🐝",
    layout="centered"
)

# Simple header
st.title("🐝 Beekeeper Agent")
st.markdown("Ask me anything - I'll automatically detect if you need evidence discovery or content generation.")

# Query Input
user_query = st.text_input("Your question:")

# Send Button
if st.button("Send") and user_query.strip():
    with st.spinner("Processing..."):
        try:
            response = requests.post(
                f"{FASTAPI_URL}/chat/stream",
                json={"query": user_query},
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                # Use a single placeholder for all content
                response_placeholder = st.empty()
                all_content = []
                
                # Fix SSE parsing with larger chunk_size to handle big "data:" lines
                for raw in response.iter_lines(decode_unicode=True, chunk_size=2048):
                    if not raw:
                        all_content.append("")  # blank line → paragraph break
                    else:
                        # Strip off "data:" if present, otherwise keep it
                        piece = raw.removeprefix("data: ").strip()
                        if piece == "[DONE]":
                            break
                        if piece and not piece.startswith("Error:"):
                            all_content.append(piece)
                    
                    # Update display with scrollable text area instead of code
                    full_text = "\n".join(all_content)
                    response_placeholder.text_area(
                        "Response",
                        full_text,
                        height=400,
                        max_chars=None
                    )
            else:
                st.error(f"Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Simple examples
st.markdown("---")
st.markdown("**Examples:**")
st.markdown("• **Evidence:** Find information about API performance")  
st.markdown("• **Summary:** Create a report for our analytics platform") 