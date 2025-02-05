from langchain_community.documents_loader import WebBaseLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph
from langgraph.memory import SqliteSaver
from prompt import *
import os



memory = SqliteSaver(':memory:')
docs = WebBaseLoader('').load()
text_split = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
text_split = text_split.from_documents(docs)
embedding = HuggingFaceEmbeddings(model_name='setence-transformers/mpnet-base-all-v2',
                                  model_kwargs={'device':'cuda'})
vectorbase = FAISS.from_documents(docs,embedding=embedding)
retriever = vectorbase.as_retriever()

llm = ChatOllama(model ='llama3.1:latest',temperature=0)
