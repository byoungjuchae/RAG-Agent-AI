from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.load import loads, dumps
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
import os

os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"]  = "CRAG"
os.environ["LANGSMITH_TRACING"] = "true"


def sentences(documents:str):

    docs = []
    doc_splits = documents.split('\n')
    for doc in doc_splits:

        if doc == '':
            pass
        else:
            docs.append(doc)
    return docs
def get_unique_union(documents: list[list]):
    
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    
    unique_docs = list(set(flattened_docs))
    
    return [loads(doc) for doc in unique_docs]
def multiple_query(vectorstore):
    
    prompt_text = """ You are an AI research assistant. You create the ten sentences similar to the {question}. only respond the ten sentences answer.
    not include the Here are ten possible anwers.
    Here is the question:
    {question}
    
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    llm = ChatOllama(model='llama3.2:latest',temperature =0)
    multiple_chain = ({'question':RunnablePassthrough()}| prompt | llm | StrOutputParser()|sentences)
  
    multiple_query = multiple_chain | vectorstore.map() |get_unique_union
    
    return multiple_query
def loading():
    
    pdf_path = './files'
    docs = PyPDFDirectoryLoader(pdf_path).load()
    text_split = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500,chunk_overlap=250)
    doc = text_split.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    retriever = FAISS.from_documents(doc,embedding=embedding).as_retriever()
    
    return retriever



if __name__ == '__main__':
    question = "What is Reward-Guided Speculative Decoding Algorithm?"
    retriever = loading()
    llm = ChatOllama(model='llama3.2:latest',temperature=0)

    #retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever,llm=llm)

    multiple_query_response = multiple_query(retriever)

    compressor = FlashrankRerank(model='ms-marco-MiniLM-L-12-v2')
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=multiple_query_response)

   
    llm = ChatOllama(model='llama3.2:latest',temperature=0)
    prompt_text = """You are an AI assistant. You answer the question to retrieve the {context}.
    
    This is the question:
    {question}
    
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    chain = ({"context": compression_retriever,"question":RunnablePassthrough()} | prompt | llm | StrOutputParser())
    response = chain.invoke(question)
    print(response)
