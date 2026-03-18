import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-advanced'))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from retriever import get_vector_retriever
from memory import get_session_history

load_dotenv()

# Main answer prompt
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

# Rephrase prompt — rewrites follow-up questions to be self-contained
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

    # Answer chain with memory
    answer_chain = RunnableWithMessageHistory(
        ANSWER_PROMPT | llm | StrOutputParser(),
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    # Rephrase chain — no memory, receives history explicitly as input
    rephrase_chain = REPHRASE_PROMPT | llm | StrOutputParser()

    return answer_chain, rephrase_chain, retriever


def chat(question: str, session_id: str = "default") -> str:
    answer_chain, rephrase_chain, retriever = get_chat_chain()

    # Step 1 — get current session history explicitly
    history = get_session_history(session_id).messages

    # Step 2 — rephrase using history passed directly, not stored in a session
    rephrased = rephrase_chain.invoke({"question": question, "history": history})
    print(f"  [Rephrased]: {rephrased}")

    # Step 3 — retrieve using rephrased question
    docs = retriever.invoke(rephrased)
    context = format_docs(docs)

    # Step 4 — answer using history + context (memory managed automatically)
    response = answer_chain.invoke(
        {"question": question, "context": context},
        config={"configurable": {"session_id": session_id}}
    )
    return response


if __name__ == "__main__":
    session = "test-session"

    print("Testing conversation memory...\n")

    q1 = "What is my car insurance policy number?"
    print(f"You: {q1}")
    print(f"Bot: {chat(q1, session)}\n")

    q2 = "When does it expire?"
    print(f"You: {q2}")
    print(f"Bot: {chat(q2, session)}\n")

    q3 = "What does it cover?"
    print(f"You: {q3}")
    print(f"Bot: {chat(q3, session)}\n")
