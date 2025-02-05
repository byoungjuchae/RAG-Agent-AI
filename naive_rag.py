from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os


os.environ['LANGSMITH_API_KEY'] = "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"
os.environ['LANGSMITH_TRACING'] = "true"
os.environ["LANGSMITH_PROJECT"] = "CRAG"


if __name__ == '__main__':
    pdf_file = './2309.07870v3.pdf'
    docs = PyPDFLoader(pdf_file).load()
    text_split = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250)
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
    
    response = chain.invoke("What is SOP?")

    print(response)