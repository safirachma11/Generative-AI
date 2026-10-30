# 11.2 ChromaDB (Halaman 125-128)
import os
import sys
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

import chromadb
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
        if os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY").startswith("sk-proj-placeholder"):
            from openai import OpenAI
            openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            resp = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
            return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]
    except Exception:
        pass
    
    vectors = embed_model.encode(texts)
    return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)

# 2. Persistent client
db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
client = chromadb.PersistentClient(path=db_dir)

# 3. Get or create collection
collection_name = "ai_knowledge_base"
try:
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
except Exception:
    client.delete_collection(collection_name)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

documents = [
    "RAG connects LLMs to external knowledge bases at inference time.",
    "Fine-tuning adapts model weights using task-specific training data.",
    "Prompt engineering guides LLM outputs without any model training.",
    "Vector databases index embeddings for fast similarity search.",
    "Agents use LLMs to plan and execute multi-step tasks with tools.",
]
ids = [f"doc_{i}" for i in range(len(documents))]
metadatas = [{"category": "technique", "index": i} for i in range(len(documents))]

if __name__ == "__main__":
    embeddings = embed(documents)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Collection '{collection_name}' size: {collection.count()} dokumen")

    # Query similarity search
    query = "how do I add knowledge to an LLM without retraining?"
    print(f"\nQuery: '{query}'")
    q_embed = embed([query])[0]
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=3,
        include=["documents", "distances", "metadatas"],
    )
    
    print("\nTop results:")
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        print(f" [Distance: {dist:.4f}] {doc}")
        print(f"  category: {meta.get('category')}")

    # Metadata filtering
    print("\nMetadata Filtering (category == 'technique'):")
    results_filtered = collection.query(
        query_embeddings=[q_embed],
        n_results=3,
        where={"category": {"$eq": "technique"}},
        include=["documents", "distances"],
    )
    for doc, dist in zip(results_filtered["documents"][0], results_filtered["distances"][0]):
        print(f" [Distance: {dist:.4f}] {doc}")

    # Update document
    print("\nUpdate Document 'doc_0':")
    updated_doc = "RAG retrieves external documents and injects them into the LLM context."
    collection.update(
        ids=["doc_0"],
        documents=[updated_doc],
        embeddings=embed([updated_doc]),
    )
    print("Document doc_0 berhasil di-update.")

    # Delete document
    print("\nDelete Document 'doc_4':")
    collection.delete(ids=["doc_4"])
    print(f"Setelah delete 'doc_4', total collection count: {collection.count()}")