from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads
import os

os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_API_KEY'] = ""
os.environ['LANGSMITH_TRACING'] = "true"
os.environ["LANGSMITH_PROJECT"] = "CRAG"



def get_unique_union(documents: list[list]):
    
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    
    unique_docs = list(set(flattened_docs))
    
    return [loads(doc) for doc in unique_docs]
def multiple_query(question,vectorstore):
    
    prompt_text = """ You are an language assistant. You create the five sentences similar to the {question}.And
    answer contains the \n at the last. and only respond the five senteces answer.
    Here is the question:
    {question}
    
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    llm = ChatOllama(model='llama3.2:latest')
    multiple_chain = (prompt | llm | StrOutputParser()|(lambda x:x.split('\n')))
  
    multiple_query = multiple_chain | vectorstore.map() | get_unique_union
    
    return multiple_query

if __name__ == "__main__":
    
    pdf_path = './files/'
    docs = PyPDFDirectoryLoader(pdf_path).load()
    text_split = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250)
    doc = text_split.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    retriever = FAISS.from_documents(doc,embedding=embedding).as_retriever()
    llm = ChatOllama(model='llama3.2:latest')
    question = "What is SOP?"
    prompt_text = """ You are an AI assistant. you must answer the {question}. you retrieve the {context}.
    
    Here is the question:
    {question}
    
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    multiple_query_response = multiple_query(question,retriever)
    
    chain_answer = ({"context":multiple_query_response,"question":RunnablePassthrough()} | prompt | llm | StrOutputParser())
    print(chain_answer.invoke(question))
    
