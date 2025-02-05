from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path
import shutil
import io



app = FastAPI()
UPLOAD_DIRECTORY = Path(__file__).parent / "uploaded_files"
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

@app.post('/')
def post(file : UploadFile= File(...), question: str = Form(...)):
    
    # To save the upload pdf file in server. This is the reason for to open pdf file.
    # If the pdf file does not save in server, you could not use it. then, you don't make a vectorbase. 
    file_path = UPLOAD_DIRECTORY/file.filename
    with open(file_path,"wb") as buffer:
        
        shutil.copyfileobj(file.file,buffer)
    
        
    docs = PyPDFLoader(file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs = text_splitter.split_documents(docs)


    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                      model_kwargs={'device':'cuda'})
    
    vectorbase = FAISS.from_documents(docs,embedding=embedding)
    retriever = vectorbase.as_retriever()
    data = retriever.get_relevant_documents(question)
    llm = ChatOllama(model='llama3.1:latest',temperature=0)
    
    prompt = PromptTemplate(template="""You are an AI engineer. And you retrieve the {document}""",input_variables=['document','question'])
    
    chain =  prompt | llm | StrOutputParser()
    return chain.invoke({"document":data,"question":question})
    

    
