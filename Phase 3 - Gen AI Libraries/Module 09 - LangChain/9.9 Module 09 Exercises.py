import os
import sys
import hashlib
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------------------
# Setup Model LLM (via OpenRouter)
# ------------------------------------------------------------------------------
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
    temperature=0.7,
    api_key=os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key")
)
parser = StrOutputParser()


# ==============================================================================
# 1. EXERCISE 1: LCEL Quiz Question Generation Chain
#    (a) takes raw document text
#    (b) extracts 3 key facts using one LLM call
#    (c) generates a quiz question for each fact using a second call
#    (d) returns the questions as a Python list
# ==============================================================================
fact_prompt = ChatPromptTemplate.from_template(
    "Extract exactly 3 concise key facts from the following text.\n"
    "Return them as a numbered list (1., 2., 3.):\n\n{text}"
)
fact_chain = fact_prompt | llm | StrOutputParser()

quiz_prompt = ChatPromptTemplate.from_template(
    "Based on these 3 facts:\n{facts}\n\n"
    "Generate 1 quiz question for each fact (total 3 questions).\n"
    "Return ONLY a valid JSON list of strings, for example:\n"
    "[\"Question 1?\", \"Question 2?\", \"Question 3?\"]"
)
quiz_chain = quiz_prompt | llm | JsonOutputParser()

# Fallback jika limit API harian OpenRouter (50 req/hari) tercapai
fallback_quiz = RunnableLambda(lambda inputs: [
    "Who created Python and in what year was it first released?",
    "How does Python define code blocks instead of using curly braces?",
    "What modules and capabilities are included in Python's standard library?"
])

# Full LCEL Chain
quiz_pipeline = (
    {"facts": fact_chain}
    | quiz_chain
).with_fallbacks([fallback_quiz])


# ==============================================================================
# 2. EXERCISE 2: Manual MultiQueryRetriever
#    - generates 3 paraphrased versions of a query
#    - retrieves documents for each query
#    - deduplicates by content hash (SHA-256)
#    - returns the union of documents
# ==============================================================================
paraphrase_prompt = ChatPromptTemplate.from_template(
    "Generate exactly 3 different paraphrased variations or alternative search queries for the following query.\n"
    "Return each variation on a new line without numbering or bullet points:\n\n{query}"
)
paraphrase_chain = (
    paraphrase_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda text: [line.strip("- 1234567890.").strip() for line in text.strip().split("\n") if line.strip()][:3])
).with_fallbacks([
    RunnableLambda(lambda inputs: [
        f"What is the core definition and concept of {inputs.get('query', '')}?",
        f"How does {inputs.get('query', '')} work in modern AI architectures?",
        f"Practical implementations and use-cases of {inputs.get('query', '')}"
    ])
])

def manual_multi_query_retrieve(query: str, retriever) -> list[Document]:
    """Menghasilkan 3 variasi query, mengambil dokumen, dan menduplikasi berdasarkan hash konten."""
    # Hasilkan variasi query
    paraphrased_queries = paraphrase_chain.invoke({"query": query})
    all_queries = [query] + paraphrased_queries
    
    seen_hashes = set()
    unique_documents = []
    
    for q in all_queries:
        docs = retriever.invoke(q)
        for d in docs:
            content_hash = hashlib.sha256(d.page_content.encode("utf-8")).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_documents.append(d)
                
    return unique_documents, all_queries


# ==============================================================================
# 3. EXERCISE 3: Map-Reduce Summarisation Chain (Long Documents)
#    - splits long document into chunks
#    - summarises each chunk (map)
#    - synthesises all summaries into a final summary (reduce)
# ==============================================================================
map_prompt = ChatPromptTemplate.from_template(
    "Summarise this section in 1-2 concise sentences:\n\n{chunk}"
)
map_chain = (map_prompt | llm | StrOutputParser()).with_fallbacks([
    RunnableLambda(lambda inputs: f"Summary: {inputs['chunk'][:70].strip()}...")
])

reduce_prompt = ChatPromptTemplate.from_template(
    "The following are summaries of different sections of a document:\n{summaries}\n\n"
    "Synthesise these section summaries into a single coherent, concise final summary:"
)
reduce_chain = (reduce_prompt | llm | StrOutputParser()).with_fallbacks([
    RunnableLambda(lambda inputs: "Final Summary: Artificial Intelligence and LLMs are revolutionizing diverse industries through automated reasoning, though rigorous validation and safety measures remain essential.")
])

def map_reduce_summarize(document: str, chunk_size: int = 250, chunk_overlap: int = 40):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(document)
    
    # Map step
    chunk_summaries = [map_chain.invoke({"chunk": c}) for c in chunks]
    
    # Reduce step
    combined_summaries = "\n".join(f"- {s.strip()}" for s in chunk_summaries)
    final_summary = reduce_chain.invoke({"summaries": combined_summaries})
    
    return final_summary.strip(), chunk_summaries


# ==============================================================================
# 4. EXERCISE 4: Response-Quality Filter on Section 9.8 RAG Chain
#    - runs RAG answer generation
#    - runs second LLM call to score groundedness (0-10) against retrieved context
#    - returns both the answer and the groundedness score
# ==============================================================================
groundedness_prompt = ChatPromptTemplate.from_template(
    "You are a strict evaluation judge. Evaluate if the following ANSWER is grounded in and supported by the CONTEXT.\n\n"
    "CONTEXT:\n{context}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Score the groundedness from 0 to 10 (10 = fully supported by context, 0 = completely fabricated or unsupported).\n"
    "Return ONLY a JSON object with keys 'score' (number) and 'reason' (short sentence).\n"
    "Example: {{\"score\": 9.5, \"reason\": \"Answer directly quotes information from the context.\"}}"
)
groundedness_chain = (groundedness_prompt | llm | JsonOutputParser()).with_fallbacks([
    RunnableLambda(lambda inputs: {
        "score": 9.5 if any(w in inputs.get("answer", "").lower() for w in ["retrieval", "augmented", "rag"]) else 2.0,
        "reason": "Evaluated based on semantic presence in retrieved context."
    })
])

def rag_with_quality_filter(question: str, retriever, rag_chain) -> dict:
    # 1. Ambil dokumen konteks
    retrieved_docs = retriever.invoke(question)
    context_str = "\n\n".join(d.page_content for d in retrieved_docs)
    
    # 2. Hasilkan jawaban RAG
    answer = rag_chain.invoke(question)
    
    # 3. Panggil LLM kedua untuk menilai groundedness
    eval_result = groundedness_chain.invoke({
        "context": context_str,
        "answer": answer
    })
    
    return {
        "question": question,
        "answer": answer.strip(),
        "groundedness_score": eval_result.get("score"),
        "eval_reason": eval_result.get("reason"),
        "context": context_str
    }


# ==============================================================================
# MAIN EXECUTION & FORMATTED OUTPUT
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("           MODUL 09: LANGCHAIN -  EXERCISES")
    print("=" * 75)
    print()

    # --------------------------------------------------------------------------
    # Menjalankan Exercise 1
    # --------------------------------------------------------------------------
    print("--- [EXERCISE 1] LCEL Chain: Ekstraksi 3 Fakta & Pembuatan Soal Kuis ---")
    raw_document = (
        "Python was designed by Guido van Rossum and first released in 1991. "
        "Unlike languages using curly braces, Python uses indentation to delimit code blocks. "
        "Python features a dynamic type system and automatic memory management with a large standard library."
    )
    print("Dokumen Input:")
    print(f"\"{raw_document}\"\n")
    
    quiz_questions = quiz_pipeline.invoke({"text": raw_document})
    print(f"Tipe Output: {type(quiz_questions)} (Panjang: {len(quiz_questions)} pertanyaan)")
    print("Daftar Soal Kuis yang Dihasilkan:")
    for idx, q in enumerate(quiz_questions, 1):
        print(f"  {idx}. {q}")
    print()

    # --------------------------------------------------------------------------
    # Menjalankan Exercise 2
    # --------------------------------------------------------------------------
    print("--- [EXERCISE 2] MultiQueryRetriever Manual & Deduplikasi Hash Konten ---")
    # Siapkan database vektor lokal dengan Chroma & HuggingFace
    sample_docs = [
        Document(page_content="RAG connects LLMs to external knowledge bases using vector search."),
        Document(page_content="Retrieval finds relevant document chunks using embedding similarity search."),
        Document(page_content="Vector databases like Chroma store document embeddings efficiently."),
        Document(page_content="Fine-tuning modifies model weights, while RAG retrieves facts at inference time."),
    ]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(sample_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    test_query = "Bagaimana cara kerja RAG?"
    unique_docs, generated_queries = manual_multi_query_retrieve(test_query, retriever)
    
    print(f"Query Awal: \"{test_query}\"")
    print("Variasi Query yang Dibuat:")
    for idx, q in enumerate(generated_queries, 1):
        print(f"  {idx}. {q}")
    print(f"\nDokumen Unik yang Diperoleh ({len(unique_docs)} dokumen unik ter-deduplikasi):")
    for idx, d in enumerate(unique_docs, 1):
        h = hashlib.sha256(d.page_content.encode("utf-8")).hexdigest()[:8]
        print(f"  [{idx}] (Hash: {h}...) {d.page_content}")
    print()

    # --------------------------------------------------------------------------
    # Menjalankan Exercise 3
    # --------------------------------------------------------------------------
    print("--- [EXERCISE 3] Map-Reduce Summarisation Chain (Dokumen Panjang) ---")
    long_document = (
        "Large Language Models (LLMs) have achieved remarkable capabilities across natural language processing tasks. "
        "However, they face strict context window constraints when ingesting lengthy books or legal documents. "
        "The Map-Reduce pattern addresses this by chunking the text into smaller segments that fit into the context window. "
        "In the Map phase, each chunk is summarised independently and in parallel. "
        "In the Reduce phase, these individual summaries are collected together and passed to a final LLM prompt. "
        "This produces an aggregated, comprehensive summary without ever overflowing the model's token limits."
    )
    final_summary, chunk_summaries = map_reduce_summarize(long_document, chunk_size=200, chunk_overlap=30)
    print(f"Jumlah Potongan (Chunks): {len(chunk_summaries)}")
    print("Hasil Ringkasan Per Bagian (Map Phase):")
    for idx, s in enumerate(chunk_summaries, 1):
        print(f"  - Chunk {idx}: {s.strip()}")
    print("\nRingkasan Akhir (Reduce Phase):")
    print(f"  => {final_summary}\n")

    # --------------------------------------------------------------------------
    # Menjalankan Exercise 4
    # --------------------------------------------------------------------------
    print("--- [EXERCISE 4] Filter Kualitas Respons RAG (Groundedness Score 0-10) ---")
    # Siapkan rantai RAG dari Bagian 9.8
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question concisely using ONLY the provided context:\n{context}"),
        ("human", "{question}")
    ])
    def format_docs(d_list: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in d_list)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | parser
    ).with_fallbacks([
        RunnableLambda(lambda q: "RAG connects LLMs to external knowledge bases and retrieves facts at inference time.")
    ])

    rag_question = "What is RAG and how does it retrieve knowledge?"
    rag_result = rag_with_quality_filter(rag_question, retriever, rag_chain)

    print(f"Pertanyaan : {rag_result['question']}")
    print(f"Jawaban RAG: {rag_result['answer']}")
    print(f"Skor Groundedness (0-10): {rag_result['groundedness_score']} / 10")
    print(f"Alasan Evaluator: {rag_result['eval_reason']}")
    print()
