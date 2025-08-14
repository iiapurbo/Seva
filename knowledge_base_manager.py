# knowledge_base_manager.py
import chromadb
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import config

class BookKnowledgeBase:
    """
    Handles all interactions with the book's vector database.
    This class is responsible for loading the database and performing hybrid search.
    """
    def __init__(self, db_path: str = config.BOOK_DB_PATH, model_name: str = "all-MiniLM-L6-v2"):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Knowledge base not found at '{db_path}'. Please run vdb_builder.py first.")
        
        print("📚 Loading Book Knowledge Base...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("autism_book_chapters")
        self.embedding_model = SentenceTransformer(model_name)

        # Load BM25 index for keyword search
        with open(os.path.join(db_path, "bm25_index.pkl"), "rb") as f:
            self.bm25_index = pickle.load(f)
        with open(os.path.join(db_path, "bm25_doc_ids.pkl"), "rb") as f:
            self.bm25_doc_ids = pickle.load(f)
        print("✅ Book Knowledge Base is ready.")

    def search(self, query: str, n_results: int = 1) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search (semantic + keyword) to find the most relevant chapter.
        """
        print(f"🔍 Searching book for: '{query}'")
        # 1. Semantic (Vector) Search
        query_embedding = self.embedding_model.encode(query).tolist()
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=10,  # Retrieve more candidates for re-ranking
            include=['metadatas', 'distances']
        )

        # 2. Keyword (BM25) Search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # 3. Hybrid Re-ranking
        candidate_scores = {}
        
        # Add vector results to candidates
        for i, doc_id in enumerate(vector_results['ids'][0]):
            similarity_score = 1 - vector_results['distances'][0][i]
            candidate_scores[doc_id] = candidate_scores.get(doc_id, 0) + (similarity_score * 0.6) # 60% weight

        # Add BM25 results to candidates
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1
        for i, doc_id in enumerate(self.bm25_doc_ids):
            bm25_score = bm25_scores[i] / max_bm25 # Normalize
            candidate_scores[doc_id] = candidate_scores.get(doc_id, 0) + (bm25_score * 0.4) # 40% weight
            
        if not candidate_scores:
            return []

        # Sort and get top N results
        sorted_candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        top_ids = [doc_id for doc_id, score in sorted_candidates[:n_results]]
        
        if not top_ids:
            return []
            
        # Retrieve full documents for the top IDs
        final_results = self.collection.get(ids=top_ids, include=['documents', 'metadatas'])
        
        # Format the output
        formatted_results = []
        for i in range(len(final_results['ids'])):
            formatted_results.append({
                "id": final_results['ids'][i],
                "text": final_results['documents'][i],
                "metadata": final_results['metadatas'][i]
            })
            
        return formatted_results