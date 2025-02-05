from langchain.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.load import loads, dumps
from operator import itemgetter
from langchain_community.document_loaders import WebBaseLoader
import bs4
question = "What is task decomposition for LLM agents?"
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")
        )
    ),
)
blog_docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 300,
    chunk_overlap =50
)
splits = text_splitter.split_documents(blog_docs)


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',model_kwargs={'device':'cuda'})
vectorstore = FAISS.from_documents(splits,embedding=embedding).as_retriever()


template = """  You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (prompt_rag_fusion | ChatOllama(model='llama3.1:latest',temperature=0) | StrOutputParser() | (lambda x : x.split("\n")))


def reciprocal_rank_fusion(results: list[list], k=60):
    
    fused_scores = {}
    for docs in results :
        
        for rank, doc in enumerate(docs):
            
            doc_str = dumps(doc)
            
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                
            previous_score = fused_scores[doc_str]
            
            fused_scores[doc_str] +=1 / (rank + k)
            
    reranked_results = [
        (loads(doc),score)
        for doc, score in sorted(fused_scores.items(), key= lambda x: x[1], reverse=True)
    ]
    
    return reranked_results

retrieval_chain_rag_fusion = generate_queries | vectorstore.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question":question})


template = """ Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context":retrieval_chain_rag_fusion,
     "question":itemgetter("question")}
    | prompt 
    | ChatOllama(model='llama3.1:latest',temperature=0)
    | StrOutputParser()
)

response = final_rag_chain.invoke({"question":question})
print(response)