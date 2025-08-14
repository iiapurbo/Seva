# streamlit_test_app.py

import streamlit as st
import requests
import json

# --- Configuration ---
# This must match the address of your chatbot FastAPI server (app.py)
API_URL = "http://127.0.0.1:8000/chat/stream"

# For simplicity, we hardcode a child_id. In a real app, this would be dynamic.
CHILD_ID = "child_01" 

# --- Streamlit App ---

st.title("Minimal Streaming Chatbot Test")
st.write(f"This app tests the streaming functionality for `child_id: {CHILD_ID}`.")
st.write("Ensure your FastAPI server is running before you start.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat box
if prompt := st.chat_input("Ask a question..."):
    # 1. Add user's message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare for the assistant's streaming response
    with st.chat_message("assistant"):
        # Create a placeholder for the streaming response
        response_placeholder = st.empty()
        full_response = ""
        
        # Prepare the data payload for the API
        payload = {"child_id": CHILD_ID, "query": prompt}
        headers = {"Content-Type": "application/json"}

        try:
            # 3. Make the streaming POST request
            with requests.post(API_URL, data=json.dumps(payload), headers=headers, stream=True) as response:
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                
                # 4. Iterate over the streaming response chunks
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        # Update the placeholder with the accumulating text and a "cursor"
                        response_placeholder.markdown(full_response + "▌")
            
            # Once the stream is done, update the placeholder one last time without the cursor
            response_placeholder.markdown(full_response)

        except requests.exceptions.RequestException as e:
            full_response = f"Error connecting to the API: {e}"
            response_placeholder.error(full_response)
        except Exception as e:
            full_response = f"An unexpected error occurred: {e}"
            response_placeholder.error(full_response)

    # 5. Add the final, complete assistant response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})