from langgraph.graph import StateGraph, add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import add_conditional_edges, ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import argparse
import os 

os.environ["TAVILIY_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = 'false'
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "agent1"
os.environ['LANGCHAIN_API_KEY'] = ''

parser = argparse.ArgumentParser()
parser.add_argument('--pdf_path',type=str,default='https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/')
args = parser.parse_args()

class State(TypedDict):
    
    messages = Annotated[list,add_messages]


class RLLM():
    
    def __init__(self,args):
        
        
        
        docs = WebBaseLoader(args.pdf_path)
        self.tool = TavilySearchResults(max_results=2)
        tools = [tool]
        llm = ChatOllama(model='llama3.1:70b',temperature=0)
        self.llm_with_tools = llm.bind_tools(tools)
        splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
        docs = docs.load_and_split(splitter)

        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                        model_kwargs={'device':'cuda'})
        vectorstores = FAISS.from_documents(docs,embedding=embedding)
        self.retriever = vectorstores.as_retriever()


        system =  "You are an AI enginner,and you retrieve the vectorstore first and then use the tools"
        self.prompt = ChatPromptTemplate.from_messages([
                                                ("system",system),
                                                ("human","{question}"),])
        self.memeory = SqliteSaver.from_conn_string(":memory:")
        self.graph_building()
    def __call__(self):
        final_state = graph.invoke(
            {"messages": [HumanMessage(content="what is the weather in seoul today?")]},
            config={"configurable": {"thread_id": 42}}
        )
        final_state["messages"][-1].content
    
        
        
    def graph_building(self):
        def chatbot(state : State):
            
            return {'messages':[self.llm_with_tools.invoke(state["messages"])]}
            
        too = ToolNode(self.tool)
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot",chatbot)
        graph_builder.add_node("tools",self.tool)
        graph_builder.add_conditional_edges("chatbot",tools_condition)
        graph_builder.add_edge("tools","chatbot")
        graph_builder.set_entry_point("chatbot")
        self.graph = graph_builder.compile(checkpointer=self.memory)


app = RLLM(args)
print(app())
