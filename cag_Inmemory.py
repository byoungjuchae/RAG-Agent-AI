from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import FastAPI
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph
from pydantic import BaseModel,Field
from langgraph.prebuilt import create_react_agent   
import os 
import uuid
import random
import time
import argparse

app = FastAPI()


llm = ChatOllama(model='llama3.2:latest')

doc = PyPDFDirectoryLoader('./files').load() 
text_split = RecursiveCharacterTextSplitter(chunk_size=250)
doc_ = text_split.split_documents(doc)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                model_kwargs={'device':'cuda'})
retriever = Chroma.from_documents(doc_,embedding=embedding).as_retriever()
prompt_text = """You are a good responser. you can say about the question.
            and you retrieve this {context}

            Here is the question:
            {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": embedding,
    }
)
user_id = "123"
namespace = (user_id, "memories")
memory_id = str(uuid.uuid4())  
memory = "I like apples"
store.put(namespace, '01', memory)
store.put(namespace,'02','I like terminator.')

@app.post('/CAG')
async def cag(query : str):
    start_time = time.time()
    if query:
        result = await store.asearch(namespace,query=query,limit=1)
        score = result[0].score
        if score >0.3:
            response = result[0].value
            print('Refer to the Memory')
            print('time:{:.4f}'.format(time.time()-start_time))
            return response
        
    chain = ({'context':retriever,'question':RunnablePassthrough()}| prompt | llm | StrOutputParser())
    key = random.random()
    response = await chain.ainvoke(query)
    print('LLM Response')
    print('time:{:.4f}'.format(time.time()-start_time))
    await store.aput(namespace,key,response)

    return response
    
    
    
    







