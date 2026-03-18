import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-advanced'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chatbot-memory'))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from retriever import get_vector_retriever
from memory import get_session_history, clear_session
from guardrails import is_allowed, is_faithful

load_dotenv()

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful insurance policy assistant.
Answer questions based only on the retrieved context.
If you don't know the answer, say "I couldn't find that in your policy documents."
Always mention which document you found the answer in."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", """Context:
{context}

Question: {question}""")
])

REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given a conversation history and a follow-up question,
rephrase the follow-up question to be self-contained.
If it's already self-contained, return it as-is.
Return only the rephrased question, nothing else."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('filename', 'unknown')} | Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def get_chat_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = get_vector_retriever()

    answer_chain = RunnableWithMessageHistory(
        ANSWER_PROMPT | llm | StrOutputParser(),
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )
    rephrase_chain = REPHRASE_PROMPT | llm | StrOutputParser()

    return answer_chain, rephrase_chain, retriever


def chat(question: str, session_id: str = "default") -> str:
    # Step 1 — guardrail: block off-topic questions
    if not is_allowed(question):
        return "I can only answer questions about your insurance policy documents. Please ask something related to your policies."

    answer_chain, rephrase_chain, retriever = get_chat_chain()

    # Step 2 — rephrase using history
    history = get_session_history(session_id).messages
    rephrased = rephrase_chain.invoke({"question": question, "history": history})

    # Step 3 — retrieve chunks
    docs = retriever.invoke(rephrased)
    context = format_docs(docs)

    # Step 4 — generate answer
    response = answer_chain.invoke(
        {"question": question, "context": context},
        config={"configurable": {"session_id": session_id}}
    )

    # Step 5 — faithfulness check: catch hallucinations
    if not is_faithful(response, context):
        return "I couldn't find reliable information in your policy documents to answer that confidently."

    return response


if __name__ == "__main__":
    session = "test-session"

    tests = [
        "What is my car insurance policy number?",
        "When does it expire?",
        "What is the capital of France?",   # should be blocked
        "What does my health policy cover?",
    ]

    for q in tests:
        print(f"\nYou: {q}")
        print(f"Bot: {chat(q, session)}")
