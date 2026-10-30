import os
import sys
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

import chromadb
from sentence_transformers import SentenceTransformer

# 1. Model embedding lokal
cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots")
if os.path.exists(cache_base) and os.listdir(cache_base):
    model_path = os.path.join(cache_base, os.listdir(cache_base)[0])
else:
    model_path = "all-MiniLM-L6-v2"

embed_model = SentenceTransformer(model_path)

def embed(texts: list[str]) -> list[list[float]]:
    vectors = embed_model.encode(texts)
    return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)

# Exercise 1: Multi-collection search helper
def search_multi_collections(client, collection_names: list[str], query_embedding: list[float], top_k: int = 5):
    all_results = []
    for name in collection_names:
        col = client.get_collection(name)
        res = col.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "distances"])
        if res["documents"] and res["distances"]:
            for doc, dist in zip(res["documents"][0], res["distances"][0]):
                all_results.append((doc, dist, name))
    all_results.sort(key=lambda x: x[1])
    return all_results[:top_k]

# Exercise 2: Tenant Vector Store (Multi-tenancy isolation)
class TenantVectorStore:
    def __init__(self, client):
        self.client = client

    def get_tenant_collection(self, tenant_id: str):
        return self.client.get_or_create_collection(f"tenant_{tenant_id}")


if __name__ == "__main__":
    print("MODULE 11 EXERCISES\n")

    client = chromadb.EphemeralClient()

    # Exercise 1
    tech_col = client.create_collection("tech_docs", metadata={"hnsw:space": "cosine"})
    business_col = client.create_collection("business_docs", metadata={"hnsw:space": "cosine"})

    tech_texts = [
        "Python is popular for machine learning and artificial intelligence.",
        "PostgreSQL with pgvector supports vector embeddings search.",
    ]
    business_texts = [
        "Q3 financial revenue grew by 15% year over year.",
        "Customer retention strategies focused on enterprise subscription plans.",
    ]

    tech_col.add(ids=["t1", "t2"], documents=tech_texts, embeddings=embed(tech_texts))
    business_col.add(ids=["b1", "b2"], documents=business_texts, embeddings=embed(business_texts))

    query_query = "What language is used for machine learning models?"
    q_vec = embed([query_query])[0]

    print(f"Query: '{query_query}'")
    multi_results = search_multi_collections(
        client,
        collection_names=["tech_docs", "business_docs"],
        query_embedding=q_vec,
        top_k=2,
    )
    for doc, dist, col_name in multi_results:
        print(f" [{col_name}] (Dist: {dist:.4f}) {doc}")

    # Exercise 2
    print()
    store = TenantVectorStore(client)
    col_a = store.get_tenant_collection("company_a")
    col_b = store.get_tenant_collection("company_b")

    doc_a = ["Company A internal security guidelines."]
    doc_b = ["Company B confidential financial report."]

    col_a.add(ids=["a1"], documents=doc_a, embeddings=embed(doc_a))
    col_b.add(ids=["b1"], documents=doc_b, embeddings=embed(doc_b))

    print(f"Tenant 'company_a' collection count: {col_a.count()} (Hanya data Company A)")
    print(f"Tenant 'company_b' collection count: {col_b.count()} (Hanya data Company B)")