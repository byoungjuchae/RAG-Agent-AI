from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from typing import Literal
from langchain.agents import create_openai_functions_agent
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "agent1"
os.environ["LANGCHAIN_API_KEY"] = ""


#tool = TavilySearchResults()
loader = PyPDFDirectoryLoader('/home/dexter/mlops/Langchain/folder')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)

docs = loader.load_and_split(text_splitter)

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                        model_kwargs = {'device':'cuda'}
                                        )

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

doc_func = lambda x: x.page_content
docs = list(map(doc_func, docs))

vectorstore = FAISS.from_texts(docs,embedding=embedding_model)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description='When you have to retrieve the documents, use it.'
)


#tools = [get_weather]

tools = [retriever_tool]

llm = ChatOllama(model='llama3:latest',temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI engineer, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_functions_agent(llm,tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(agent_executor.invoke({"input":"What is the meaning of Alpha-CLIP?"}))
#agent_executor = AgentExecutor(agent=agent,tools=tools,handle_parsing_erros=True, verbose=False)

