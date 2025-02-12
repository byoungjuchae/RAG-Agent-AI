from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os


os.environ['LANGSMITH_API_KEY'] = ""
os.environ['LANGSMITH_TRACING'] = "true"
os.environ["LANGSMITH_PROJECT"] = "CRAG"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"


if __name__ == '__main__':
    pdf_file = './files'
    docs = PyPDFDirectoryLoader(pdf_file).load()
    text_split = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500,chunk_overlap=250)
    docs = text_split.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    vectorstore = FAISS.from_documents(docs,embedding=embedding).as_retriever()
    llm = ChatOllama(model='llama3.2:latest')
    prompt_text = """ You are an AI assistant. you must answer the {question}. you retrieve the {context}.
    
    Here is the question:
    {question}
    
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = ({"context":vectorstore,"question":RunnablePassthrough()}| prompt | llm | StrOutputParser())
    
    response = chain.invoke("What is Reward-Guided Speculative Decoding Algorithm?",temperature=0)

    print(response)
