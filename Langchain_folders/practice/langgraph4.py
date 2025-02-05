from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Annotated
from langchain_community.chat_models import ChatOllama
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START,END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
class State(TypedDict):
    
    messages : Annotated[list, add_messages]


def chatbot(state: State):
    
    return {"messages":[llm.invoke(state["messages"])]}

tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOllama(model='llama3.1:latest',temperature=0)
import pdb
pdb.set_trace()
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools=[tool])
graph_builder = StateGraph(State)
memory = SqliteSaver.from_conn_string(":memory:")
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tool",tool_node)
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tool","chatbot")

graph = graph_builder.compile(checkpoint=memory)