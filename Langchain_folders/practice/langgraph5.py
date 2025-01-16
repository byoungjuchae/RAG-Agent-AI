from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain.tools.retriever import create_retriever_tool
loader = PyPDFDirectoryLoader("/")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
loader = loader.load_and_split(splitter)

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                        model_kwargs={'device':'cuda'})
vectorbase = FAISS.from_documents(loader,embedding=embedding_model)
retriever = vectorbase.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description='When you have to retrieve the documents, use it.'
)

tools = [retriever_tool]
llm = ChatOllama(model='llama3.1:latest',temperature=0)

llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI engineer.answer the question and to retrieve the tools",
        ),
        ("human", "{input}"),
    ]
)
chain = prompt | llm_with_tools | StrOutputParser()

print(chain.invoke({"input":"What is the ALpha-CLIP?"}))