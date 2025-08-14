import fitz  # PyMuPDF
import nltk
import json
import pickle
import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --- Configuration ---
# <<< MODIFICATION: Updated PDF filename to match the file you provided for the new system >>>
SOURCE_PDF_PATH = "Turn_Autism_Around_-_PhD_Mary_Lynch_Barbera.pdf"

# <<< MODIFICATION: Updated DB_PATH to match the new system's config.py >>>
DB_PATH = "./autism_book_db"
COLLECTION_NAME = "autism_book_chapters"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000  # Note: This is now less relevant as we chunk by chapter
CHUNK_OVERLAP = 200

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF file using PyMuPDF for better layout preservation."""
    print(f"📖 Reading text from '{pdf_path}'...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Source PDF not found at '{pdf_path}'. Please make sure it's in the project root.")
    
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(tqdm(doc, desc="Extracting PDF pages")):
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text

def split_into_chapters(full_text: str) -> list[tuple[int, str]]:
    """Splits the full text into chapters based on a regex pattern."""
    print("Slicing text into chapters...")
    text_with_markers = "CHAPTER 0\n" + full_text
    
    # This regex pattern is robust for this specific book's format
    split_text = re.split(r'(CHAPTER \d+)', text_with_markers)
    
    chapters = []
    for i in range(1, len(split_text), 2):
        chapter_title_line = split_text[i].strip()
        chapter_content = split_text[i+1].strip()
        
        match = re.search(r'\d+', chapter_title_line)
        chapter_number = int(match.group(0)) if match else (i // 2)

        if chapter_number == 0:
            chapters.append((0, chapter_content))
        else:
            if chapters and chapters[0][0] == 0:
                preface_content = chapters.pop(0)[1]
                chapter_content = preface_content + "\n\n" + chapter_content
            chapters.append((chapter_number, chapter_content))
            
    print(f"Found {len(chapters)} potential chapter sections.")
    return chapters

def process_and_chunk_by_chapter(chapters: list[tuple[int, str]]) -> list[dict]:
    """Processes each chapter as a single chunk to preserve context."""
    print("Processing full chapters as individual chunks...")
    
    all_chunks = []
    # Simplified chapter titles for better matching, based on the Table of Contents.
    chapter_titles = {
        1: "Early Signs of Autism Are an Emergency",
        2: "Is It Autism, ADHD, or “Just” a Speech Delay?",
        3: "Keep Your Child Safe",
        4: "An Easy Assessment to Figure Out Your Starting Point",
        5: "Gather Materials and Make a Plan",
        6: "Stop the Tantrums and Start the Learning",
        7: "Develop Social and Play Skills",
        8: "Teach Talking and Following Directions",
        9: "Talking but Not Conversational",
        10: "Solve Picky Eating",
        11: "Solving Sleep Issues",
        12: "Dispose of the Diapers",
        13: "Desensitize Doctor, Dentist, and Haircut Visits",
        14: "Become Your Child’s Best Teacher and Advocate"
    }
    
    for chapter_num, chapter_text in tqdm(chapters, desc="Processing chapters"):
        if 1 <= chapter_num <= 14: # Only process the 14 main chapters
            all_chunks.append({
                "text": chapter_text,
                "metadata": {
                    "chapter_number": chapter_num,
                    "chapter_title": chapter_titles.get(chapter_num, f"Chapter {chapter_num}")
                }
            })
            
    print(f"Created {len(all_chunks)} text chunks, one for each main chapter.")
    return all_chunks

def create_bm25_index(documents: list, doc_ids: list, db_path: str):
    """Creates and saves a BM25 index for keyword search."""
    print("Tokenizing corpus for BM25 keyword index...")
    tokenized_corpus = [doc.lower().split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save the BM25 index
    with open(Path(db_path) / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
        
    # <<< MODIFICATION: CRITICAL CHANGE FOR NEW SYSTEM COMPATIBILITY >>>
    # The new `knowledge_base_manager.py` expects 'bm25_doc_ids.pkl', not 'chunk_ids_mapping.pkl'.
    with open(Path(db_path) / "bm25_doc_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)
    
    print("✅ BM25 index and document ID mapping saved successfully.")

def main():
    """Main function to build the vector database."""
    print("="*80)
    print("🚀 Starting Vector Database Builder for Integrated System")
    print("="*80)

    if Path(DB_PATH).exists():
        response = input(f"⚠️ Database already exists at '{DB_PATH}'.\n    Do you want to delete and rebuild it? (y/n): ").lower()
        if response != 'y':
            print("🛑 Build aborted by user.")
            return
        print(f"🔥 Deleting existing database: {DB_PATH}")
        shutil.rmtree(DB_PATH)

    os.makedirs(DB_PATH, exist_ok=True)

    # 1. Extract and process text
    raw_text = extract_text_from_pdf(SOURCE_PDF_PATH)
    chapters = split_into_chapters(raw_text)
    documents = process_and_chunk_by_chapter(chapters)
    
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    ids = [f"chapter_{doc['metadata']['chapter_number']}" for doc in documents]

    # 2. Setup ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # 3. Create embeddings and add to DB
    print(f"🧠 Generating embeddings using '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    
    print(f"✍️ Adding {len(ids)} documents to the '{COLLECTION_NAME}' collection...")
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("✅ All documents embedded and stored in ChromaDB.")

    # 4. Create the necessary BM25 index for hybrid search
    create_bm25_index(texts, ids, DB_PATH)

    print("\n" + "="*80)
    print("🎉 Vector Database build complete!")
    print(f"   Database is ready at: {DB_PATH}")
    print("   You can now run the chatbot API and the main system.")
    print("="*80)

if __name__ == "__main__":
    main()