from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

loader = PyPDFDirectoryLoader('/')

text_split = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
docs = loader.load_and_split(text_split)


embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                        model_kwargs={'device':'cuda'})
def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

doc_func = lambda x: x.page_content
docs = list(map(doc_func, docs))

vectordb = FAISS.from_texts(docs,
                            embedding=embedding_model)

retriever = vectordb.as_retriever(search_type="similarity_score_threshold",
                                 search_kwargs={"score_threshold": 0.5},)


prompt = PromptTemplate.from_template("""
                                      You are an AI Engineer. you have to answer the question through retrieving the {context}.
                                      
                                      Question:
                                      {question}
                                      """)

llm = ChatOllama(model='gemma:latest',temperatures=0)

chain = {'context':retriever,'question':RunnablePassthrough()} | prompt | llm | StrOutputParser()

print(chain.invoke('What is the feature of Pixel Aligned Language Models?'))