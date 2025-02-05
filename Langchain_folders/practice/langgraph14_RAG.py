from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain import hub
from langgraph.graph import StateGraph
from typing import Annotated
from typing_extensions import TypedDict

import os


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "agent1"


docs = PyPDFDirectoryLoader("")
text_split = RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=0)
docs = docs.split_documents(text_split)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                  model_kwargs={"device":"cuda"})
vectorstore = FAISS.from_documents(docs,embedding=embedding)
retreiver = vectorstore.as_retriever()

class Stategraph(TypedDict):
    
    question : str
    genereation : str
    documents : List[str]
    

def retrieve(state):
    
    print('----RETREIVE----')
    question = state['question']
    
    
    documents = retreiver.get_relevant_documents(state['question'])
    
    return {'documents': documents, 'question':question}

def generate(state):

    print("----GENERATE----")
    question = state['question']
    document = state['document']
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOllama(model='llama3.1:latest',temperature=0)
    
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context":docs,"question":question})
    
    return {'question':question, 'document':document, 'generation':generation}    

def hallucination(state):
    llm = ChatOllama(model='llama3.1:latest',format='json',temperature=0)
    
    prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)
    hallucination_grader = prompt | llm | JsonOutputParser()
    score = hallucination_grader.invoke({"documents":docs,"generation":generation})
    
    grade = score['score']
    
    return {}
    
    

def retrieve_grade(state):
    
    print('----RETRIEVE GRADE----')
    llm = ChatOllama(model='llama3.1:latest',format='json',temperature=0)
    question = state['question']
    documents = state['documents']
    prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)   
    grader_chain = prompt | llm | JsonOutputParser()
    
    filtered_docs = []
    for d in documents:
        score = grader_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

