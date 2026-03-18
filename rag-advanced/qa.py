import os
import glob
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retriever import get_vector_retriever, get_hybrid_retriever, get_metadata_filtered_retriever, add_reranking

load_dotenv()

DOCS_PATH = "rag-basics/docs"

PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions about insurance policy documents.
Use ONLY the context below to answer the question. If the answer is not in the context,
say "I couldn't find that information in the provided documents."
Always mention which document and page you found the answer on.

Context:
{context}

Question: {question}

Answer:
"""


def load_documents():
    # Load all PDFs into memory for BM25 (keyword search)
    # BM25 works on raw documents, not a vector DB
    documents = []
    pdf_files = glob.glob(f"{DOCS_PATH}/*.pdf")
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception:
            pass

    # Split into chunks — same as ingest.py
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', doc.metadata.get('filename', 'unknown'))}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def build_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, mode: str = "hybrid", doc_type: str = None, rerank: bool = False):
    """
    mode: "vector" | "hybrid" | "filtered"
    doc_type: only used when mode="filtered" e.g. "endorsement" or "policy"
    rerank: if True, adds Cohere reranking on top of the retriever
    """
    print(f"\nMode: {mode} | Rerank: {rerank} | Question: {question}")

    if mode == "vector":
        retriever = get_vector_retriever()
    elif mode == "hybrid":
        documents = load_documents()
        retriever = get_hybrid_retriever(documents)
    elif mode == "filtered":
        retriever = get_metadata_filtered_retriever(doc_type or "policy")

    # Wrap retriever with reranker if requested
    if rerank:
        retriever = add_reranking(retriever)

    chain = build_chain(retriever)
    answer = chain.invoke(question)

    print(f"\nAnswer: {answer}")

    # Show sources
    chunks = retriever.invoke(question)
    print("\nSources:")
    seen = set()
    for doc in chunks:
        source = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}:{page}"
        if key not in seen:
            seen.add(key)
            print(f"  - {source} (page {page})")


if __name__ == "__main__":
    # Test 1 — vector search without reranking
    ask("What is the car insurance premium?", mode="vector", rerank=False)

    # Test 2 — same question with reranking — compare the sources order
    ask("What is the car insurance premium?", mode="vector", rerank=True)

    # # Test 3 — filtered search (only endorsement documents)
    # ask("What's the policy details?", mode="filtered", doc_type="endorsement")
