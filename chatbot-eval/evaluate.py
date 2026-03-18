import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-advanced'))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from retriever import get_vector_retriever

load_dotenv()

# Eval set — 10 questions about your policy documents with expected answers
# These are ground truth answers you know are correct
EVAL_SET = [
    {
        "question": "What is the car insurance policy number?",
        "ground_truth": "POPMCAR00100997384"
    },
    {
        "question": "When does the car insurance policy expire?",
        "ground_truth": "20/09/2025"
    },
    {
        "question": "What is the registration number of the car?",
        "ground_truth": "25-BH-2978J"
    },
    {
        "question": "What is the make and model of the insured car?",
        "ground_truth": "Hyundai Creta 1.5 S (O) IVT Petrol"
    },
    {
        "question": "What is the seating capacity of the insured vehicle?",
        "ground_truth": "5"
    },
    {
        "question": "What health insurance product is mentioned in Policy Copy 11?",
        "ground_truth": "Health Companion"
    },
    {
        "question": "What is the hospitalization coverage in the health policy?",
        "ground_truth": "Upto Sum Insured"
    },
    {
        "question": "How many days of pre-hospitalization expenses are covered?",
        "ground_truth": "30 days"
    },
    {
        "question": "What is the ambulance cover amount in the health policy?",
        "ground_truth": "Rs. 3000"
    },
    {
        "question": "What is the No Claim Bonus benefit in the health policy?",
        "ground_truth": "Enhancement of Sum Insured by 20% of expiring Base Sum Insured, maximum up to 100%"
    },
]


def build_eval_dataset():
    retriever = get_vector_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"Running RAG on {len(EVAL_SET)} questions...")
    for i, item in enumerate(EVAL_SET):
        question = item["question"]
        print(f"  [{i+1}/{len(EVAL_SET)}] {question}")

        # Retrieve chunks
        docs = retriever.invoke(question)
        context_texts = [doc.page_content for doc in docs]

        # Generate answer
        context_str = "\n\n".join(context_texts)
        prompt = f"""Answer based only on the context below.
Context: {context_str}
Question: {question}
Answer:"""
        response = llm.invoke(prompt)

        questions.append(question)
        answers.append(response.content)
        contexts.append(context_texts)
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })


def run_evaluation():
    dataset = build_eval_dataset()

    # Wrap LangChain LLM and embeddings for RAGAS
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )

    def avg(val):
        # Results can be a float or a list of floats depending on RAGAS version
        if isinstance(val, list):
            return sum(v for v in val if v is not None) / len([v for v in val if v is not None])
        return val

    print("\n── RAGAS Scores ──────────────────────────")
    print(f"  Faithfulness:      {avg(results['faithfulness']):.3f}  (target: >0.80)")
    print(f"  Answer Relevancy:  {avg(results['answer_relevancy']):.3f}  (target: >0.80)")
    print(f"  Context Precision: {avg(results['context_precision']):.3f}  (target: >0.70)")
    print(f"  Context Recall:    {avg(results['context_recall']):.3f}  (target: >0.70)")
    print("──────────────────────────────────────────")

    return results


if __name__ == "__main__":
    run_evaluation()
