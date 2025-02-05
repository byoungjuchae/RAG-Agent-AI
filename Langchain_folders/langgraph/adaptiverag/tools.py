from langchain_community.from_documents import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.tavily_search_api import TavilySearchResults


def RAG(foldername,query):
    
    docs = PyPDFDirectoryLoader(filename).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                      model_kwargs={'device':'cuda'})
    vectorbase = FAISS.from_documents(docs,embedding=embedding)
    retriever = vectorbase.as_retriever()
    
    response = retriever.get_relevant_documents(query)
    
    return response


def search(query):
    
    search = TavilySearchResults(k=3)
    
    return search 
    