from langgraph.graph import END, StateGraph, MessagesState,START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.chat_models import ChatOllama



llm = ChatOllama(model='llama3.1:latest',temperature=0)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)
graph = graph_builder.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)