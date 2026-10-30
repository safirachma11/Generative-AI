import os
import sys
import numpy as np
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from sentence_transformers import SentenceTransformer

# 1. Inisialisasi model embedding lokal
cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots")
if os.path.exists(cache_base) and os.listdir(cache_base):
    model_path = os.path.join(cache_base, os.listdir(cache_base)[0])
else:
    model_path = "all-MiniLM-L6-v2"

embed_model = SentenceTransformer(model_path)

def embed(texts: list[str]) -> list[list[float]]:
    try:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key and not openai_key.startswith("sk-proj-placeholder"):
            from openai import OpenAI
            openai_client = OpenAI(api_key=openai_key)
            resp = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
            return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]
    except Exception:
        pass
    vectors = embed_model.encode(texts)
    return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)

if __name__ == "__main__":
    documents = [
        {"id": "d1", "text": "RAG connects LLMs to external knowledge bases."},
        {"id": "d2", "text": "Fine-tuning updates model weights on task-specific data."},
        {"id": "d3", "text": "ChromaDB is an open-source local vector database."},
    ]

    query_text = "how to add external knowledge to an LLM?"
    print(f"Query: '{query_text}'\n")

    pinecone_key = os.environ.get("PINECONE_API_KEY")

    if pinecone_key and not pinecone_key.startswith("your_"):
        try:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=pinecone_key)
            INDEX_NAME = "ai-knowledge-base"

            sample_dim = len(embed(["test"])[0])
            existing_indexes = [idx.name for idx in pc.list_indexes()]

            if INDEX_NAME not in existing_indexes:
                print(f"Membuat index baru: {INDEX_NAME} (Dimensi: {sample_dim})...")
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=sample_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            
            index = pc.Index(INDEX_NAME)

            vectors = [
                {
                    "id": doc["id"],
                    "values": embed([doc["text"]])[0],
                    "metadata": {"text": doc["text"], "source": "guide"},
                }
                for doc in documents
            ]
            index.upsert(vectors=vectors)
            print("Index Stats:\n", index.describe_index_stats())

            q_vector = embed([query_text])[0]
            results = index.query(
                vector=q_vector,
                top_k=3,
                include_metadata=True,
            )

            print("\nTop Matches (Live Pinecone Cloud):")
            for match in results["matches"]:
                print(f"[{match['score']:.4f}] {match['metadata']['text']}")

        except Exception as e:
            print(f"[Catatan Koneksi Pinecone]: {e}")
    else:
        print("[Info] PINECONE_API_KEY tidak ditemukan di .env.")
        print("[Menjalankan Simulasi Komputasi Vector Search Secara Lokal]:\n")

        doc_vectors = np.array(embed([d["text"] for d in documents]))
        q_vector = np.array(embed([query_text])[0])

        dot_product = np.dot(doc_vectors, q_vector)
        norm_docs = np.linalg.norm(doc_vectors, axis=1)
        norm_q = np.linalg.norm(q_vector)
        scores = dot_product / (norm_docs * norm_q)

        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        print("Top Matches (Local Cosine Similarity):")
        for score, doc in ranked:
            print(f"[{score:.4f}] {doc['text']}")