from langgraph.graph import StateGraph, add_messages
from langchain_ollama import ChatOllama
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langchain_core.messages import HumanMessage

os.environ["TAVILIY_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "agent1"
os.environ['LANGCHAIN_API_KEY'] = ''

tool = TavilySearchResults(max_results=2)

memory = SqliteSaver.from_conn_string(":memory:")
class State(TypedDict):
    
    messages : Annotated[list, add_messages]
    

llm = ChatOllama(model='llama3.1:70b',temperature=0)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    
    return {'messages':[llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot",chatbot)

tools = ToolNode([tool])
graph_builder.add_node("tools",tools)
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools","chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)

final_state = graph.invoke(
    {"messages": [HumanMessage(content="what is the weather in seoul today?")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content