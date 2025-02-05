from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


db = PyPDFDirectoryLoader('/')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

db = db.load_and_split(text_splitter)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                   model_kwargs={'device':'cuda'})
vectorstore = FAISS.from_documents(db,
                                   embedding=embeddings) 

retriever = vectorstore.as_retriever(search_type='similarity_score_threshold',
                                     search_kwargs={"score_threshold":0.5},)

prompt = PromptTemplate.from_template("""
                                      

                                      """)

llm = ChatOllama(model ='llama3:latest',temperature=0)
conversation = ConversationChain(llm=llm,memory=ConversationBufferMemory())
conversation("I'm doing well! Just having a conversation with an AI.")
