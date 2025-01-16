from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, add_conditional_edges
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.checkpoint.sqlite import SqliteSaver
from langchain.output_parsers import JsonOutputParser
from langgraph.messages import add_messages
from typing_extensions import TypedDict
from typing import List
import os


os.environ["TAVILY_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "agent1"
os.environ["LANGCHAIN_API_KEY"] = ''


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)
embeddingmodel = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                       model_kwargs='cuda')
vectorstore = FAISS.from_documents(doc_splits,embedding=embeddingmodel)
retriever = vectorstore.as_retriever()

tool = TavilySearchResults(max_result=2)
memory = SqliteSaver.from_conn_string(":memory:")

llm = ChatOllama(model='llama3.1:latest',format="json",temperature=0)
llm_with_tools = llm.bind_tools([tool])


promptTemplate = ChatPromptTemplate("""
                                    You are an AI enginner in Google.And you can answer my {question}. Also you retrieve the vectorbase and web surfing""")


class State(TypedDict):
    
    message : Annoated[list,add_messages]
    
    
def chatbot(state: State):
    
    return {'messages':[llm_with_tools.invoke(state['messages'])]}

graph_build = StateGraph(State)
graph_build.add_node("chatbot",chatbot)
graph_build.add_conditional_edges("chatbot",[tools])
graph_build.add_node("tools",tool)
graph_build.add_edge("tools","chatbot")
graph_build.set_entry_point("chatbot")
graph = graph_build.compile(checkpointer=memory)






