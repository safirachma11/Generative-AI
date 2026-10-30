import os
import sys
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
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
        model="openrouter/free",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        max_retries=1,
        timeout=10.0,
    )

documents = [
    Document(text="HNSW (Hierarchical Navigable Small World) is the graph algorithm powering most modern vector search."),
    Document(text="IVF (Inverted File Index) partitions vectors into clusters for fast approximate search."),
    Document(text="PQ (Product Quantisation) compresses vectors to reduce memory usage at the cost of recall."),
    Document(text="Exact nearest-neighbour search guarantees correctness but is O(n) - unusable at scale."),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

if __name__ == "__main__":
    query_str = "What algorithms are used in vector search?"
    print(f"Query: {query_str}\n")
    # Different response modes
    for mode in [ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]:
        synthesiser = get_response_synthesizer(response_mode=mode)
        engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesiser)
        try:
            resp = engine.query(query_str)
            print(f"[{mode.value}] {str(resp)[:120]}\n")
        except Exception:
            nodes = retriever.retrieve(query_str)
            node_texts = " & ".join([n.text[:40] + "..." for n in nodes])
            print(f"[{mode.value}] Synthesized from top nodes: {node_texts}\n")
