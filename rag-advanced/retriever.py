import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "ai-applications"


def get_vector_retriever(documents=None):
    # Vector retriever — finds semantically similar chunks using embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})


def get_hybrid_retriever(documents):
    # BM25 retriever — keyword based search (like a search engine)
    # Works on raw documents in memory, not a vector DB
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    # Vector retriever — semantic search using embeddings
    vector_retriever = get_vector_retriever()

    # EnsembleRetriever — combines BM25 + vector with weighted scores
    # weights=[0.4, 0.6] means: 40% BM25, 60% vector
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    return hybrid_retriever


def get_metadata_filtered_retriever(doc_type: str):
    # Filter retrieval to only search within a specific document type
    # e.g. doc_type="endorsement" only searches endorsement documents
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return vector_store.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"must": [{"key": "metadata.doc_type", "match": {"value": doc_type}}]}
        }
    )


def add_reranking(retriever, top_n: int = 4):
    # CohereRerank re-orders the retrieved chunks by true relevance
    # top_n=4: after reranking, keep only the top 4 chunks
    # We first retrieve more chunks (k=10 or 20) then rerank down to top_n
    reranker = CohereRerank(
        model="rerank-v3.5",
        top_n=top_n
    )

    # ContextualCompressionRetriever wraps any retriever with a reranker
    # base_retriever fetches candidates, compressor re-orders and filters them
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever
    )


if __name__ == "__main__":
    # Test vector retrieval only (no documents needed)
    print("Testing vector retriever...")
    retriever = get_vector_retriever()
    chunks = retriever.invoke("What is the policy number?")
    print(f"Found {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} | {chunk.metadata.get('filename')} | Page {chunk.metadata.get('page')} ---")
        print(chunk.page_content[:200])
        print()
