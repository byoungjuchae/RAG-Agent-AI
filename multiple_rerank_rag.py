from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.load import loads, dumps
import os

os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"]  = "CRAG"
os.environ["LANGSMITH_TRACING"] = "true"



def get_unique_union(documents: list[list]):
    
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    
    unique_docs = list(set(flattened_docs))
    
    return [loads(doc) for doc in unique_docs]
def multiple_query(vectorstore):
    
    prompt_text = """ You are an language assistant. You create the five sentences similar to the {question}.And
    answer contains the \n at the last. and only respond the five senteces answer.
    Here is the question:
    {question}
    
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    llm = ChatOllama(model='llama3.2:latest')
    multiple_chain = ({'question':RunnablePassthrough()}| prompt | llm | StrOutputParser()|(lambda x:x.split('\n')))
  
    # multiple_query = multiple_chain | vectorstore.map() 
    
    return multiple_chain
def loading():
    
    pdf_path = './files/2309.07870v3.pdf'
    docs = PyPDFLoader(pdf_path).load()
    text_split = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250)
    doc = text_split.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    retriever = FAISS.from_documents(doc,embedding=embedding).as_retriever()
    
    return retriever

def re_rank(results,k=60):
    
    fused_scores = {}
    for docs in results :
        
        for rank, doc in enumerate(docs):
            
            doc_str = dumps(doc)
            
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            
            previous_score = fused_scores[doc_str]
            
            fused_scores[doc_str] +=1 / (rank+k)
    
    reanked_results = [
        (loads(doc),score)
        for doc, score in sorted(fused_scores.items(), key= lambda x:x[1], reverse=True)
    ]
    
    return reanked_results    

if __name__ == '__main__':
    question = "What is SOP?"
    retriever = loading()
    multiple_query_response = multiple_query(retriever)
    
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=retriever)
    re_ranking = multiple_query_response | compression_retriever.map() 
   
    llm = ChatOllama(model='llama3.2:latest')
    prompt_text = """You are an AI assistant. You answer the {question} to retrieve the {context}.
    
    This is the question:
    {question}
    
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    chain = ({"context": re_ranking,"question":RunnablePassthrough()} | prompt | llm | StrOutputParser())
    response = chain.invoke("What is SOP?")
    print(response)