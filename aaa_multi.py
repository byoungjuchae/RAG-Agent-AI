import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import os
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
from sentence_transformers import SentenceTransformer, util
from langchain.smith import RunEvalConfig


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "RAG"
os.environ["LANGSMITH_API_KEY"] = ""


loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")
        )
    ),
)
blog_docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 300,
    chunk_overlap =50
)
splits = text_splitter.split_documents(blog_docs)


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',model_kwargs={'device':'cuda'})
vectorstore = FAISS.from_documents(splits,embedding=embedding).as_retriever()


from langchain.prompts import ChatPromptTemplate

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOllama(model='llama3.1:latest',temperature=0)
generate_chain = (prompt | llm | StrOutputParser() | (lambda x : x.split("\n")))


from langchain.load import dumps, loads


def get_unique_union(documents: list[list]):
    
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    
    unique_docs = list(set(flattened_docs))
    
    return [loads(doc) for doc in unique_docs]

question = "What is task decomposition for LLM agents?"

retrieval_chain = generate_chain | vectorstore.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})

print(len(docs))

def answering(inputs: dict):
    from operator import itemgetter

    template = """
    Answer the following question based on this context:
    {context}

    Question : {question}

    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model="llama3.1:latest",temperature=0)

    final_rag_chain = (
        {"context":retrieval_chain,
        "question":itemgetter("question")
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )

    response = final_rag_chain.invoke({"question":question})
    return {"answer":response}



def semscore_evaluator(run: Run, example: Example) -> dict:
    # 출력값과 정답 가져오기
    student_answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("answer", "")

    # SentenceTransformer 모델 로드
    model = SentenceTransformer("all-mpnet-base-v2")

    # 문장 임베딩 생성
    student_embedding = model.encode(student_answer, convert_to_tensor=True)
    reference_embedding = model.encode(reference_answer, convert_to_tensor=True)

    # 코사인 유사도 계산
    cosine_similarity = util.pytorch_cos_sim(
        student_embedding, reference_embedding
    ).item()

    return {"key": "sem_score", "score": cosine_similarity}


# cot_qa_evaluator = LangChainStringEvaluator(
#     "cot_qa",
#     config={"llm": ChatOllama(model="llama3.2:latest", temperature=0)},
#     prepare_data=lambda run, example: {
#         "prediction": run.outputs["answer"],
#         "reference": run.outputs["context"],
#         "input": example.inputs["question"],
#     },
# )

heuristic_evaluators = [
    semscore_evaluator,answer_relevancy
]
dataset_name = "DBRX"

experiment_results = evaluate(
    
    answering,
    data=dataset_name,
    evaluators = heuristic_evaluators,
    experiment_prefix = "test-dbrx"

)