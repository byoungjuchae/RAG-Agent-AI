from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI, Form
import os


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "agent1"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""



app = FastAPI()


@app.post('/')
async def web_search(question : str):
    
    tool = TavilySearchResults(max_results=2)
    llm = ChatOllama(model='llama3.1:latest',temperature=0)
    llm_with_tool = llm.bind_tools([tool])
    prompt = PromptTemplate(template= """ You are an expert all of the fields.So, you can answer the {question} with search the Web.""",
                            input_variables=['question'])
    chain = prompt | llm_with_tool | StrOutputParser()
    
    return chain.invoke({'question':question})

    
    
    
    
    
    
    