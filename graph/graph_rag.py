from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain
import os



user = "neo4j"
password = "1532587@"

question = "What is SOP?"

graph = Neo4jGraph(url='bolt://localhost:7687',username=user,password=password)

chain = GraphCypherQAChain.from_llm(
    ChatOllama(model='llama3.2:latest',temperature=0),allow_dangerous_requests=True, graph=graph, verbose=True
)
chain.invoke("Who played in Top Gun?")
