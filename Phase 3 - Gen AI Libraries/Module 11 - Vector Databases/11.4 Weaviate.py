import os
import sys
import numpy as np
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from sentence_transformers import SentenceTransformer

# 1. Embedding lokal
cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots")
if os.path.exists(cache_base) and os.listdir(cache_base):
    model_path = os.path.join(cache_base, os.listdir(cache_base)[0])
else:
    model_path = "all-MiniLM-L6-v2"

embed_model = SentenceTransformer(model_path)

# 2. Dokumen sampel
documents = [
    {"id": "doc1", "text": "Weaviate is an open-source vector database with hybrid search.", "category": "database"},
    {"id": "doc2", "text": "BM25 is a keyword-based ranking algorithm used in traditional information retrieval.", "category": "search"},
    {"id": "doc3", "text": "Dense vector search uses neural embeddings to capture semantic meaning.", "category": "ai"},
]

def hybrid_search(query: str, alpha: float = 0.5, top_k: int = 2):
    """
    Simulasi Hybrid Search Weaviate:
    Score = (alpha * vector_score) + ((1 - alpha) * keyword_score)
    """
    # Vector Search (Cosine Similarity)
    doc_vectors = np.array(embed_model.encode([d["text"] for d in documents]))
    q_vector = np.array(embed_model.encode([query])[0])
    dot_product = np.dot(doc_vectors, q_vector)
    norm_docs = np.linalg.norm(doc_vectors, axis=1)
    norm_q = np.linalg.norm(q_vector)
    vector_scores = dot_product / (norm_docs * norm_q)

    # Keyword Search (Simple Token Match / BM25 Proxy)
    query_tokens = set(query.lower().split())
    keyword_scores = []
    for d in documents:
        doc_tokens = set(d["text"].lower().split())
        match_count = len(query_tokens.intersection(doc_tokens))
        keyword_scores.append(match_count / max(len(query_tokens), 1))
    keyword_scores = np.array(keyword_scores)

    # Normalisasi & Penggabungan
    combined_scores = (alpha * vector_scores) + ((1 - alpha) * keyword_scores)
    ranked_indices = np.argsort(combined_scores)[::-1][:top_k]

    return [(combined_scores[i], vector_scores[i], keyword_scores[i], documents[i]) for i in ranked_indices]

if __name__ == "__main__":
    query = "vector search open-source database"
    alpha = 0.5
    print(f"Query: '{query}'")
    print(f"Alpha: {alpha} (0.0 = Pure BM25, 1.0 = Pure Vector)\n")

    results = hybrid_search(query, alpha=alpha, top_k=3)
    print("Hasil Hybrid Search:")
    for score, v_score, k_score, doc in results:
        print(f"[Total Score: {score:.4f} | Dense: {v_score:.4f} | BM25: {k_score:.4f}]")
        print(f"  {doc['text']}\n")