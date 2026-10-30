# 10.6 Summary Index (Halaman 120-122)
import os
import sys
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from llama_index.core import SummaryIndex, Document, Settings
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

# 3. Data Dokumen terdistribusi per section
sections = [
    Document(text="Introduction: This report analyses the performance of five LLMs on coding, reasoning, and language tasks."),
    Document(text="Section 1 - Coding: Claude Sonnet 4.5 scored highest on HumanEval with 94.2%. GPT-4o followed at 90.1%."),
    Document(text="Section 2 - Reasoning: GPT-4o led on GSM8K with 97.1%. Claude Sonnet 4.5 achieved 96.8%."),
    Document(text="Section 3 - Language: Both models performed similarly on MMLU, within 1% of each other."),
    Document(text="Conclusion: For coding tasks, prefer Claude Sonnet 4.5. For balanced performance, GPT-4o is the safer choice."),
]

# 4. Bangun SummaryIndex
index = SummaryIndex.from_documents(sections)
query_engine = index.as_query_engine()

if __name__ == "__main__":
    prompt = "Which model should I use for a coding project?"
    print(f"Query: {prompt}\n")
    try:
        response = query_engine.query(prompt)
        print("Response (LLM generated):\n", response)
    except Exception as e:
        print(f"[Peringatan API] Gagal memanggil LLM: {e}\n")
        print("Response (Fallback Manual):")
        print("Based on the report, for coding tasks, prefer Claude Sonnet 4.5 (scored highest on HumanEval with 94.2%).")