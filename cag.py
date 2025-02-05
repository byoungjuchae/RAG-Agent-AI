from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
import json
import pandas as pd
import os

os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"]  = "CRAG"
os.environ["LANGSMITH_TRACING"] = "true"



def docs_page(docs):
    docs_doc=[]
 
    for doc_ in docs:
        
        
        docs_doc.append(Document(doc_.page_content))
        
    return docs_doc
pdf_path = './2309.07870v3.pdf'
doc = PyPDFLoader(pdf_path).load()
text_split = RecursiveCharacterTextSplitter(chunk_size=250)
docs = text_split.split_documents(doc)
re_docs = docs_page(docs)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
vectorstore = FAISS.from_documents(docs,embedding=embedding)


retriever = vectorstore.as_retriever()
prompt_text = """ You are a AI assistant.You must make the 50 questions and answers. You make the answers to retrieve the {context}.
and make the questions about the answer.

Here is the {context}.  The format is following json format:
'question':question, 'answer': answer
"""
prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOllama(model='llama3.2:latest',format='json',temperature=0)

chain = ({'context':retriever} | prompt | llm | JsonOutputParser())
response = chain.invoke("You make the question and answer.")
with open('./cag_file.json', 'w') as f:
    json.dump(response, f)


