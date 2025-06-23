#!/usr/bin/env python
"""
🐝 Beekeeper Agent - Simple Chat UI
"""
import streamlit as st
import requests

# Configuration
FASTAPI_URL = "http://localhost:8003"

# Page configuration
st.set_page_config(
    page_title="Beekeeper Agent",
    page_icon="🐝",
    layout="centered"
)

# Simple header
st.title("Beekeeper Agent")

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
                # Use a single placeholder that gets updated
                response_placeholder = st.empty()
                all_content = []
                
                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        
                        if data_str == "[DONE]":
                            break
                        
                        if data_str and not data_str.startswith("Error:"):
                            all_content.append(data_str)
                            
                            # Update the single placeholder with all content
                            full_text = "\n\n".join(all_content)
                            response_placeholder.text(full_text)
                            
            else:
                st.error(f"Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Simple examples
st.markdown("---")
st.markdown("**Examples:**")
st.markdown("• *Evidence:* Find information about API performance")  
st.markdown("• *Summary:* Create a report for our analytics platform") 