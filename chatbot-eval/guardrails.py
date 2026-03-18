import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Topics our chatbot is allowed to answer
ALLOWED_TOPICS = """
- Insurance policies (health, car, life, home)
- Policy numbers, dates, premiums
- Coverage details and exclusions
- Claims process
- Policy documents and terms
"""

GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"""You are a topic classifier for an insurance policy assistant.
Your job is to determine if a user question is related to insurance policies or not.

Allowed topics:
{ALLOWED_TOPICS}

Respond with ONLY one word:
- "allowed" if the question is about insurance policies
- "blocked" if the question is off-topic"""),
    ("human", "{question}")
])

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a faithfulness checker for an AI assistant.
Your job is to verify if an answer is grounded in the provided context.
Be lenient — if the answer is mostly supported by the context, mark it as faithful.
Only mark as unfaithful if the answer contains clear fabrications not in the context at all.

Respond with ONLY one word:
- "faithful" if the answer is mostly supported by the context
- "unfaithful" if the answer contains clear fabrications not present in the context"""),
    ("human", """Context:
{context}

Answer:
{answer}

Is this answer faithful to the context?""")
])


def is_allowed(question: str) -> bool:
    """Returns True if question is on-topic, False if off-topic."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = GUARDRAIL_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().lower()
    return result == "allowed"


def is_faithful(answer: str, context: str) -> bool:
    """Returns True if answer is grounded in context, False if hallucinating."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = FAITHFULNESS_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"answer": answer, "context": context}).strip().lower()
    return result == "faithful"


if __name__ == "__main__":
    print("Testing guardrails...\n")

    questions = [
        "What is my car insurance policy number?",   # allowed
        "What does my health policy cover?",          # allowed
        "Write me a poem about cats",                 # blocked
        "What is the capital of France?",             # blocked
        "How do I file a claim?",                     # allowed
    ]

    for q in questions:
        allowed = is_allowed(q)
        status = "✅ allowed" if allowed else "❌ blocked"
        print(f"{status}: {q}")
