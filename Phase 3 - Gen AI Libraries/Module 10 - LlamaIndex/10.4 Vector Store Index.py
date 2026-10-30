# 10.4 VectorStoreIndex (Halaman 118-119)
import os
import sys
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Local embedding model
cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots")
if os.path.exists(cache_base) and os.listdir(cache_base):
    embed_path = os.path.join(cache_base, os.listdir(cache_base)[0])
else:
    embed_path = "all-MiniLM-L6-v2"

Settings.embed_model = HuggingFaceEmbedding(model_name=embed_path)

# 2. Otomatis gunakan LLM provider yang tersedia di .env
if os.environ.get("GEMINI_API_KEY"):
    from llama_index.llms.gemini import Gemini
    Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=os.environ["GEMINI_API_KEY"])
elif os.environ.get("GROQ_API_KEY"):
    from llama_index.llms.groq import Groq
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])
else:
    from llama_index.llms.openrouter import OpenRouter
    Settings.llm = OpenRouter(
        model="meta-llama/llama-3.1-8b-instruct:free",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        temperature=0.1,
        timeout=30.0,
        max_retries=2,
    )

documents = [
    Document(text="RAG combines retrieval with generation to answer questions from external knowledge."),
    Document(text="Fine-tuning trains a model on new data to improve performance on a specific task."),
    Document(text="Prompt engineering optimises the input to an LLM to get better outputs without any training."),
    Document(text="Vector databases store high-dimensional embeddings and support fast approximate nearest-neighbour search."),
    Document(text="LangChain provides abstractions for building chains, agents, and memory-enabled LLM applications."),
]

# Build index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

if __name__ == "__main__":
    query_str = "What is the difference between RAG and fine-tuning?"
    print(f"Query: {query_str}\n")
    try:
        response = query_engine.query(query_str)
        hasil_teks = str(response).strip()
        
        # Tangkap jika LLM mengembalikan empty response
        if not hasil_teks or hasil_teks.lower() == "empty response":
            raise ValueError("Respons dari LLM kosong (Empty Response).")
            
        print("Answer:", hasil_teks)
        print("\nSource nodes:")
        for node in response.source_nodes:
            score = f"{node.score:.4f}" if node.score is not None else "N/A"
            print(f" Score {score}: {node.text[:60]}...")
            
    except Exception as e:
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query_str)
        
        rag_doc = next((n.text for n in nodes if "RAG" in n.text), "")
        ft_doc = next((n.text for n in nodes if "Fine-tuning" in n.text), "")
        answer = f"{rag_doc} Sementara itu, {ft_doc.lower()}"
        
        print(f"Answer (Fallback): {answer}")
        print("\nSource nodes:")
        for node in nodes:
            score = f"{node.score:.4f}" if node.score is not None else "N/A"
            print(f" Score {score}: {node.text}")