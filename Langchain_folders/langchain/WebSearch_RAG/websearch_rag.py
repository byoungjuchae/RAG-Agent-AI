from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, UploadFile, Form, File
from pathlib import Path
import shutil
import os


os.environ['TAVILY_API_KEY'] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "webserach"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


app = FastAPI()
UPLOAD_DIRECTORY = Path(__file__).parent / 'uploadfiles'
UPLOAD_DIRECTORY.mkdir(parents=True,exist_ok=True)

@app.post('/')
async def web_rag(file : UploadFile =File(...), question : str = Form(...)):
    file_path = UPLOAD_DIRECTORY/file.filename
    with open(file_path,"wb") as f:
        shutil.copyfileobj(file.file,f)
    
    doc = PyPDFLoader(file_path).load()
    text_split = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    doc = text_split.split_documents(doc)
    
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                      model_kwargs={'device':'cuda'})
    vectorbase = FAISS.from_documents(doc,embedding=embedding)
    retriever = vectorbase.as_retriever()
    
    docs = retriever.get_relevant_documents(question)
    tool = TavilySearchResults(max_results=2)
    
    llm = ChatOllama(model='llama3.1:latest',temperature=0)
    llm_with_tool = llm.bind_tools([tool])
    
    prompt = PromptTemplate(template="""You are an AI engineer." You can answer the {question} to retrieve the {document}.
                            And you can use the Web surfing.""",input_variables=['question','document'])
    
    chain = prompt | llm_with_tool | StrOutputParser()
    
    return chain.invoke({'question':question,'document':docs})




