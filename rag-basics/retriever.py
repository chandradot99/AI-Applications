from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "rag-basics/chroma_db"


def get_retriever():
    # Load the existing ChromaDB (already populated by ingest.py)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Returns top 4 most relevant chunks for any question
    # search_type="mmr": Maximum Marginal Relevance — avoids returning 4 near-identical chunks
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )
    return retriever


def retrieve(question: str):
    retriever = get_retriever()
    chunks = retriever.invoke(question)

    print(f"\nQuestion: {question}")
    print(f"Found {len(chunks)} relevant chunks:\n")

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        print(f"--- Chunk {i+1} | Source: {source} | Page: {page} ---")
        print(chunk.page_content)
        print()

    return chunks


if __name__ == "__main__":
    retrieve("What is the policy number?")
