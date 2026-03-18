from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retriever import get_retriever

load_dotenv()

# Prompt template — instructs the LLM to only use the provided context
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


def format_docs(docs):
    # Combine all chunk texts into one context string
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def get_qa_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # cheap and fast, good for development
        temperature=0          # deterministic — no creativity for factual Q&A
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = get_retriever()

    # Modern LangChain chain using LCEL (LangChain Expression Language)
    # question → retriever → format_docs → prompt → llm → parse output
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(question: str):
    chain, retriever = get_qa_chain()

    # Get answer
    answer = chain.invoke(question)

    # Get source documents for citations
    source_docs = retriever.invoke(question)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")

    # Show citations
    print("\nSources:")
    seen = set()
    for doc in source_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}:{page}"
        if key not in seen:
            seen.add(key)
            print(f"  - {source} (page {page})")


if __name__ == "__main__":
    ask("What are all my car insurance details across all policies?")
