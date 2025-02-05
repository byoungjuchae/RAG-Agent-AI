from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    
    messages : Annotated[list, add_messages]

llm = ChatOllama(model='llama3.1:latest',temperature=0)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    
    return {'messages':[llm_with_tools.invoke(state['messages'])]}
    
graph_builder.add_node("chatbot",chatbot)
tool_node = ToolNode(tools)
graph_builder.add_node("tools",tool_node)
graph_builder.add_conditional_edges("chatbot",tools_condition)


graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()