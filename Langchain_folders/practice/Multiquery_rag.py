from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

llm = ChatOllama(model='gemma:latest',temperature=0)

retriever = MultiQueryRetriever.from_llm(llm=llm,retriever=vectordb.as_retriever())

prompt = PromptTemplate.from_template("""You are an AI engineer.you have to answer the question and refer the {context}.
                                      
                                      Question:{question}""")



chain = {"context": retriever | format_docs, "question":RunnablePassthrough()} | prompt | llm | StrOutputParser()

print(chain.invoke('What is Honeybee?'))
