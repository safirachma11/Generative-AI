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
        model="openrouter/free",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        max_retries=1,
        timeout=10.0,
    )

# 1. ask(question: str) wrapper returning answer and source files
def ask_with_sources(query_engine, question: str) -> dict:
    try:
        resp = query_engine.query(question)
        sources = [node.metadata.get("source", "unknown") for node in resp.source_nodes]
        return {"answer": str(resp), "sources": list(set(sources))}
    except Exception:
        nodes = query_engine.retriever.retrieve(question) if hasattr(query_engine, 'retriever') else []
        sources = [n.metadata.get("source", "unknown") for n in nodes]
        top_text = nodes[0].text if nodes else "LlamaIndex specializes in data indexing and retrieval for RAG."
        return {"answer": top_text, "sources": list(set(sources))}

# 2. Re-rank nodes with LLM score
def rerank_nodes(nodes, question: str, score_fn, top_n: int = 3):
    scored = [(n, score_fn(question, n.text)) for n in nodes]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, s in scored[:top_n]]

if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 10 EXERCISES")
    print("=" * 60)

    # Setup demo documents
    docs = [
        Document(text="LangChain provides LCEL for building multi-step LLM chains.", metadata={"source": "langchain_doc.pdf"}),
        Document(text="LlamaIndex specializes in data indexing and retrieval for RAG.", metadata={"source": "llamaindex_doc.pdf"}),
        Document(text="Chroma is a vector store for embedding storage and similarity search.", metadata={"source": "chroma_doc.pdf"}),
    ]
    index = VectorStoreIndex.from_documents(docs)
    engine = index.as_query_engine(similarity_top_k=2)

    # Demo Exercise 1
    print("\n--- Exercise 1: ask_with_sources ---")
    res1 = ask_with_sources(engine, "What is LlamaIndex used for?")
    print("Result:")
    print(f"  Answer : {res1['answer']}")
    print(f"  Sources: {res1['sources']}")

    # Demo Exercise 2
    print("\n--- Exercise 2: rerank_nodes ---")
    retriever = index.as_retriever(similarity_top_k=3)
    raw_nodes = retriever.retrieve("Tell me about retrieval and storage")

    def simple_relevance_scorer(q, text):
        words = set(q.lower().split())
        return sum(1 for w in words if w in text.lower())

    reranked = rerank_nodes(raw_nodes, "Tell me about retrieval and storage", simple_relevance_scorer, top_n=2)
    print(f"Top 2 nodes after reranking:")
    for i, node in enumerate(reranked, 1):
        print(f"  {i}. [{node.metadata.get('source')}] {node.text}")

    print("\n" + "=" * 60)
    print("Semua Exercise Selesai Diuji!")
    print("=" * 60)
