from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults

class State(TypedDict):
    
    messages : Annotated[list, add_messages]
    
def chatbot(state: State):
    return{"messages":[llm.invoke(state["messages"])]}

tool = TavilySearchResults(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools) 
llm = ChatOllama(model='llama3.1:latest',temperature=0)


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot",chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools",tool_node)
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools","chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

