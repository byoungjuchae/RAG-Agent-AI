import pandas as pd
from langsmith import Client
import os
from langchain import hub
from langchain_ollama.chat_models import ChatOllama
from langsmith.evaluation import evaluate
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "RAG"
os.environ["LANGSMITH_API_KEY"] = "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"



# client = Client()
# inputs = [
#     "How many tokens was DBRX pre-trained on?",
#     "Is DBRX a MOE model and how many parameters does it have?",
#     "How many GPUs was DBRX trained on and what was the connectivity between GPUs?",
# ]

# outputs = [
#     "DBRX was pre-trained on 12 trillion tokens of text and code data.",
#     "Yes, DBRX is a fine-grained mixture-of-experts (MoE) architecture with 132B total parameters.",
#     "DBRX was trained on 3072 NVIDIA H100s connected by 3.2Tbps Infiniband",
# ]

# # Dataset
# qa_pairs = [{"question": q, "answer": a} for q, a in zip(inputs, outputs)]
# df = pd.DataFrame(qa_pairs)

# # Write to csv
# csv_path = "./acsv.csv"
# df.to_csv(csv_path, index=False)


# dataset_name= "DBRX"
# dataset = client.create_dataset(
#     dataset_name = dataset_name,
#     description = "QA pairs about DBRX model.",
# )

# client.create_examples(
#     inputs= [{"question":q} for q in inputs],
#     outputs=[{'answer':a} for a in outputs],
#     dataset_id = dataset.id,
# )
def predict_rag_answer(example:dict):
    llm = ChatOllama(model='llama3.1:latest')
    response = llm.invoke(example["question"])
    return {"answer":response}

grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")

def answer_evaluator(run, example) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """
    
    # Get summary
    input_question = example.inputs["question"]
    reference = example.outputs["answer"]
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatOllama(model="llama3.2:latest", temperature=0)

    # Structured prompt
    
    answer_grader = grade_prompt_answer_accuracy | llm

    # Get score
    score = answer_grader.invoke({"question": input_question,
                                  "correct_answer": reference,
                                  "student_answer": prediction})

    score = score["score"]

    return {"key": "answer_score", "score": score}


dataset_name = "DBRX"

experiment_results = evaluate(
    
    predict_rag_answer,
    data = dataset_name,
    evaluators = [answer_evaluator],
    experiment_prefix = "rag_qa_oai"
)