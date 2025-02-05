from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import START,END,StateGraph, add_messages, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from typing import Annotated
from pydantic import BaseModel
from fastapi import FastAPI
import os

# os.environ['TAVILY_API_KEY'] = ""
# os.environ['LANGCHAIN_API_KEY'] = ""
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://langsmith.com'

app = FastAPI()
llm = ChatOllama(model='llama3.1:latest',temperature=0)
memory = SqliteSaver.from_conn_string(':memory:')

def chatbot(state: MessagesState):
    
    return {'messages': [llm.invoke(state['messages'])]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node('chatbot',chatbot)
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)

@app.post('/')
def llama(question : str):
    

    final_state = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": 42}}
    )

    return final_state['messages'][-1].content


