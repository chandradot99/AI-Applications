import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_PATH = "rag-basics/docs"
CHROMA_PATH = "rag-basics/chroma_db"


def ingest():
    # 1. Load all PDFs from the docs folder, skipping encrypted ones
    print("Loading PDFs...")
    documents = []
    skipped = []
    pdf_files = glob.glob(f"{DOCS_PATH}/**/*.pdf", recursive=True) + glob.glob(f"{DOCS_PATH}/*.pdf")

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            skipped.append(pdf_path)
            print(f"  Skipped (encrypted/unreadable): {os.path.basename(pdf_path)}")

    print(f"  Loaded {len(documents)} pages from {len(pdf_files) - len(skipped)} files")
    if skipped:
        print(f"  Skipped {len(skipped)} encrypted files")

    # 2. Split documents into smaller chunks
    # chunk_size=500: each chunk is max 500 characters
    # chunk_overlap=50: chunks overlap by 50 chars so context isn't lost at boundaries
    print("Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    # 3. Embed each chunk and store in ChromaDB
    # text-embedding-3-small: cheapest OpenAI embedding model, works great for this
    if not os.getenv("OPENAI_API_KEY"):
      print("Error: OPENAI_API_KEY is not set in your .env file")

    print("Embedding and storing in ChromaDB...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        print(f"  Stored {db._collection.count()} chunks in ChromaDB at '{CHROMA_PATH}'")
        print("Done!")
    except Exception as e:
        print(f"Error during embedding: {e}")


if __name__ == "__main__":
    ingest()
