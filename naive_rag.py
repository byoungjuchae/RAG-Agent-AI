<<<<<<< HEAD
from langchain_community.document_loaders import PyPDFLoader
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
=======
from langchain_ollama.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import InMemoryByteStore
from glob import glob
import os




def pdf_load(file):

    docs = PyPDFLoader(file).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    
    return docs


def main(folder,question):
    # files = glob(os.path.join(folder,'*.pdf'))
    docs = []
    # for idx,f  in enumerate(files):
    doc = pdf_load('/home/doc/2310.03270v4.pdf')
    docs.extend(doc)
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                      model_kwargs={'device':'cuda'})
    
    #bm25_retriever = BM25Retriever()
    # if ty == "faiss":
    llm = ChatOllama(model='llama3.2:latest',temperature=0)
    vectorbase = FAISS.from_documents(docs,embedding=embedding)
    retriever = vectorbase.as_retriever(search_type='similarity')
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever,llm=llm)

    # if ty == "Multi":
    #     store = InMemoryByteStore()
    #     retriever = MultiVectorRetriever(vectorstore=vectorbase,byte_store=store)
    # if ty == "BM25" :
    #     retriever = BM25Retriever.from_documents(docs)
    #     retriever.k = 2
    # if ty == "Ensemble":
    #     vectorbase = FAISS.from_documents(docs,embedding=embedding)
    #     faiss_retriever = vectorbase.as_retriever(search_type='similarity',k=4)
    #     bm25_retriever = BM25Retriever.from_documents(docs)
    #     bm25_retriever.k = 2
    #     retriever = EnsembleRetriever(
    #         retrievers= [bm25_retriever,faiss_retriever],weights=[0.7,0.5]
    #     )
    
    # if ty == "Long_context":
        
    #retriever = vectorbase.as_retriever(search_type='similarity',k=4)
    # retriever = EnsembleRetriever(
    #     retrievers= [bm25_retriever,vectorbase],
    #     method="reciprocal_rank_fusion"
    # )
   

    re_doc = retriever_from_llm.get_relevant_documents(question)
    llm = ChatOllama(model='llama3.2:latest',temperature=0)
    prompt = PromptTemplate(template="""You are an AI engineer. And you retrieve the {document} and answer the {question} in one line.""",input_variables=['document','question'])
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"document":re_doc,"question":question})

if __name__ == '__main__':
    folder_name = '/home/dexter/mlops/doc/'
    question = "How much time does BayesDiff require, and how long does QAT and PTQ take?"
    response = main(folder_name,question=question)
>>>>>>> 58d80437d0aed88fbfb8a67e729bda1622befd74
    print(response)