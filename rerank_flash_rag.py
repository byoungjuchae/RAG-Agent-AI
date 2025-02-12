from langchain.document_loaders import PyPDFDirectoryLoader
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
    pdf_path = './files'
    load = PyPDFDirectoryLoader(pdf_path).load()
    text_split = RecursiveCharacterTextSplitter(chunk_size=250)
    docs = text_split.split_documents(load)
    
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    

    retriever = FAISS.from_documents(docs,embedding=embedding).as_retriever()
#     retriever = Chroma(
#     embedding_function=embedding,
#     persist_directory="./chroma_langchain_db",
# ).as_retriever()

    compressor = FlashrankRerank()
    compression_retreiver = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=retriever)

    prompt_text = """You are an AI assistant. You answer the {question} to retrieve the {context}.
    
    This is the question:
    {question}
    
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    llm = ChatOllama(model='llama3.2:latest',temperature=0)
    
    chain = ({'context':compression_retreiver,'question':RunnablePassthrough()} | prompt | llm | StrOutputParser())
    
    print(chain.invoke("What is Reward-Guided Speculative Decoding Algorithm?"))
    
