import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

load_dotenv()

DOCS_PATH = "rag-basics/docs"
COLLECTION_NAME = "ai-applications"

# Metadata we attach to each chunk — used for filtering in Week 2
def get_metadata(pdf_path: str) -> dict:
    filename = os.path.basename(pdf_path)
    # Detect document type from filename
    if "Endorsement" in filename:
        doc_type = "endorsement"
    elif "Policy" in filename:
        doc_type = "policy"
    else:
        doc_type = "other"

    return {
        "filename": filename,
        "doc_type": doc_type,
    }


def ingest():
    # 1. Load all PDFs, skipping encrypted ones
    print("Loading PDFs...")
    documents = []
    skipped = []
    pdf_files = glob.glob(f"{DOCS_PATH}/*.pdf")

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Add custom metadata to each page
            extra_meta = get_metadata(pdf_path)
            for doc in docs:
                doc.metadata.update(extra_meta)

            documents.extend(docs)
        except Exception:
            skipped.append(pdf_path)
            print(f"  Skipped (encrypted/unreadable): {os.path.basename(pdf_path)}")

    print(f"  Loaded {len(documents)} pages from {len(pdf_files) - len(skipped)} files")

    # 2. Split into chunks
    print("Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    # 3. Set up Qdrant collection
    print("Setting up Qdrant collection...")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )

    # Create collection if it doesn't exist
    # 1536 dimensions = size of text-embedding-3-small vectors
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"  Created collection '{COLLECTION_NAME}'")
    else:
        print(f"  Collection '{COLLECTION_NAME}' already exists")

    # Create payload indexes — required for metadata filtering in Qdrant
    # Idempotent — safe to run even if indexes already exist
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.doc_type",
        field_schema=PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.filename",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("  Created payload indexes for 'doc_type' and 'filename'")

    # 4. Embed and store in Qdrant
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set in your .env file")
        return

    print("Embedding and storing in Qdrant...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Create vector store once with a longer timeout
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )

        # Upload in small batches of 50
        BATCH_SIZE = 50
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            print(f"  Uploaded batch {i // BATCH_SIZE + 1}/{total_batches}")

        print(f"  Stored {len(chunks)} chunks in Qdrant collection '{COLLECTION_NAME}'")
        print("Done!")
    except Exception as e:
        print(f"Error during embedding: {e}")


if __name__ == "__main__":
    ingest()
