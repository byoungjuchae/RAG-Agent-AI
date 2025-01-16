from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from prompt import *
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START,END
from typing import List
from typing_extensions import TypedDict


memory = SqliteSaver.from_conn_string(':memory:')

class GraphState(TypedDict):
    
    question : str
    generation : str
    documents : List[str]

graph_flow = StateGraph(GraphState)
graph_flow.add_node("grade",grade)
graph_flow.add_node("answer",answer)
graph_flow.add_node("web_search",web_search)
graph_flow.add_node("retreiver",retriever)



graph_flow.add_conditional_edges(START,
                            query_analysis,
                            {
                                "RETREIVE": "retriever",
                                "WEB":"web_search"
                                
                            })
graph_flow.add_edge("web_search","generate")
graph_flow.add_edge("retreiver","grade")
graph_flow.add_conditional_edges("grade",
                            relevant
                            {
                                "YES": "generate"
                                "NO" : "rewrite"
                                
                            })
graph_flow.add_conditional_edges("generate",
                            hallucination
                            {
                                "NO": "answer",
                                "YES" :"generate"
                            })
graph = graph_flow.compile(checkpointer=memory)