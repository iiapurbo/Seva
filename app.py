# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
from fastapi.responses import StreamingResponse
import asyncio

import config
from chatbot_orchestrator import Chatbot
from knowledge_base_manager import BookKnowledgeBase

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multi-Child ASD Support Chatbot API",
    description="An API for a multi-source, child-specific chatbot with streaming.",
    version="1.1.0-streaming"
)

# --- Global Resources ---
# Load the knowledge base once at startup to be shared across all sessions.
try:
    book_kb_singleton = BookKnowledgeBase(db_path=config.BOOK_DB_PATH)
except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    print("The chatbot cannot start without the knowledge base. Please run the vdb_builder.py script.")
    book_kb_singleton = None

# In-memory store for chatbot sessions. Each child gets their own instance.
chat_sessions: Dict[str, Chatbot] = {}

# --- Pydantic Models for API ---
class ChatRequest(BaseModel):
    child_id: str = Field(..., example="child_01", description="The unique identifier for the child.")
    query: str = Field(..., example="How was he yesterday?", description="The parent's question for the chatbot.")

class ChatResponse(BaseModel):
    child_id: str
    query: str
    response: str

# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Main endpoint to interact with the child-specific chatbot (non-streaming).
    """
    if book_kb_singleton is None:
        raise HTTPException(status_code=503, detail="The Knowledge Base is not available. Please run the vdb_builder.py script.")

    child_id = request.child_id
    query = request.query

    # Get or create a chatbot session for the specific child
    if child_id not in chat_sessions:
        print(f"✨ Creating new chat session for child: {child_id}")
        chat_sessions[child_id] = Chatbot(child_id=child_id, book_kb=book_kb_singleton)
    
    chatbot_instance = chat_sessions[child_id]
    
    try:
        # Generate the response using the orchestrator
        response_text = chatbot_instance.generate_response(query)
        
        return ChatResponse(
            child_id=child_id,
            query=query,
            response=response_text
        )
    except Exception as e:
        print(f"❌ Unhandled error during chat for child '{child_id}': {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the chat.")

@app.post("/chat/stream")
async def handle_streaming_chat(request: ChatRequest):
    """
    New endpoint to handle chat requests with a streaming response.
    """
    if book_kb_singleton is None:
        raise HTTPException(status_code=503, detail="The Knowledge Base is not available.")

    child_id = request.child_id
    query = request.query

    # Get or create a chatbot session for the specific child
    if child_id not in chat_sessions:
        print(f"✨ Creating new stream-enabled chat session for child: {child_id}")
        chat_sessions[child_id] = Chatbot(child_id=child_id, book_kb=book_kb_singleton)
    
    chatbot_instance = chat_sessions[child_id]

    try:
        # Return a StreamingResponse, which FastAPI handles automatically.
        return StreamingResponse(
            chatbot_instance.generate_streaming_response(query),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"❌ Unhandled error during streaming chat for child '{child_id}': {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the stream.")

@app.get("/")
def root():
    return {"message": "ASD Support Chatbot API is running."}