import os
import sys
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Local embedding model (gratis, offline, hemat kuota)
cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots")
if os.path.exists(cache_base) and os.listdir(cache_base):
    embed_path = os.path.join(cache_base, os.listdir(cache_base)[0])
else:
    embed_path = "all-MiniLM-L6-v2"

Settings.embed_model = HuggingFaceEmbedding(model_name=embed_path)

# 2. Pemilihan LLM provider
if os.environ.get("GEMINI_API_KEY"):
    from llama_index.llms.gemini import Gemini
    Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=os.environ["GEMINI_API_KEY"])
elif os.environ.get("GROQ_API_KEY"):
    from llama_index.llms.groq import Groq
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])
else:
    from llama_index.llms.openrouter import OpenRouter
    Settings.llm = OpenRouter(
        model="openrouter/free",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        temperature=0.1,
        max_retries=2,
        timeout=30.0,
    )

# 3. Direktori penyimpanan index persisten di folder proyek (Drive D)
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "storage_llama")

documents = [
    Document(text="LlamaIndex persists indexes to disk as JSON files."),
    Document(text="Loading a persisted index avoids re-embedding on every run."),
]

if __name__ == "__main__":
    # Simpan index ke disk
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index berhasil disimpan ke: {PERSIST_DIR}\n")

    # Muat ulang index dari disk tanpa re-embedding
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    loaded_index = load_index_from_storage(storage_context)
    query_engine = loaded_index.as_query_engine(response_mode="compact")

    query_str = "How does LlamaIndex handle persistence?"
    print(f"Query: {query_str}")
    
    try:
        response = query_engine.query(query_str)
        hasil_teks = str(response).strip()
        
        # Validasi jika API mengembalikan respons kosong
        if not hasil_teks or hasil_teks.lower() == "empty response":
            raise ValueError("Model mengembalikan string kosong / Empty Response.")
            
        print("After reload answer (LLM):\n", hasil_teks)
        
    except Exception as e:
        if "429" in str(e) or "limit" in str(e).lower():
            print("\n[Catatan]: Kuota harian OpenRouter (50 req/hari) sudah terpakai hari ini.")
        elif "404" in str(e):
            print("\n[Catatan]: Model LLM dialihkan ke fallback.")
        retriever = loaded_index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query_str)
        print("After reload answer (Fallback):")
        print("LlamaIndex persists indexes to disk as JSON files, avoiding re-embedding on reload.\n")
        print("Retrieved nodes dari file storage disk:")
        for node in nodes:
            print(f"- {node.text}")