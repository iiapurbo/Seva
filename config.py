# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free" 
ROUTER_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free" 

# --- Database and File Paths ---
BOOK_DB_PATH = "./autism_book_db"
CHILD_DATA_BASE_DIR = "child_data"

# --- Chatbot Behavior ---
HTTP_REFERER = "http://localhost"
APP_NAME = "ASD Support Chatbot"

# --- API Communication ---
# The URL where your *other* FastAPI server (from api.py) is running.
# Running on a different port like 8001 is a good way to test.
ANOMALY_API_URL = "http://127.0.0.1:8001"