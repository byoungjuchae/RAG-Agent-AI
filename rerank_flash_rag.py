from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"]  = "CRAG"
os.environ["LANGSMITH_TRACING"] = "true"

if __name__ == '__main__':
    
    def pretty_print_docs(docs):
        print(
            f"\n{'-' * 100}\n".join(
                [
                    f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                    for i, d in enumerate(docs)
                ]
            )
        )
    pdf_path = './files/2309.07870v3.pdf'
    load = PyPDFLoader(pdf_path).load()
    text_split = RecursiveCharacterTextSplitter(chunk_size=250)
    docs = text_split.split_documents(load)
    
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    
    retriever = Chroma(
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
).as_retriever()
    #retriever = FAISS.from_documents(docs,embedding=embedding).as_retriever()

    compressor = FlashrankRerank()
    compression_retreiver = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=retriever)

    prompt_text = """You are an AI assistant. You answer the {question} to retrieve the {context}.
    
    This is the question:
    {question}
    
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    llm = ChatOllama(model='llama3.2:latest',temperature=0)
    
    chain = ({'context':compression_retreiver | pretty_print_docs,'question':RunnablePassthrough()} | prompt | llm | StrOutputParser())
    
    print(chain.invoke("What is SOP?"))
    