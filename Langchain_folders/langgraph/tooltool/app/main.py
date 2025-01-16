from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing import List
from typing_extensions import TypedDict
from langchain.agents import AgentExecutor, create_react_agent, AgentOutputParser, initialize_agent
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from langchain import hub
from langchain.memory import ConversationBufferWindowMemory
import asyncio
import cv2
from tool import *
from io import BytesIO
import os
from celery import Celery
import json
import base64
from PIL import Image

        
celery_app = Celery('task',
                    broker='http:localhost:6380/0',
                    backend ='http:localhost:6380/0')

os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "agent"
os.environ['LANGCHAIN_TRACING_V2'] = "true"


class MultiFormatOutputParser(AgentOutputParser):
    def parse(self, text: str):
        if text.startswith("data:image"):
            return self._parse_image(text)
        elif text.startswith("{") and text.endswith("}"):
            return self._parse_json(text)
        else:
            return text

    def _parse_image(self, text: str) -> Image.Image:
        base64_im = text.split(",", 1)[0] if "," in text else text
        base64_data = text.split(",", 1)[1] if "," in text else text
        image_data = base64.b64decode(base64_data)
        return base64_data

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")


app = FastAPI()
@app.post('/start/upload')
async def upload(query2: UploadFile = File(...)):
    pass
    

@app.post('/start')
async def start(query : str = Form(...)):
    memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=5)
    llm = ChatOllama(model='llama3.2:latest')
    prompt = PromptTemplate(template="""
    You are an intelligent assistant with access to the following tools: {tools}. Answer the question accurately. If your response is a base64-encoded image, stop immediately and provide it as the final answer without taking any further actions. Otherwise, use up to two actions and then provide a final answer.

    When responding, strictly follow this format:

    Question: [The question you must answer]
    Thought: [Your thoughts about what to do next]
    Action: [The action to take, one of: {tool_names}]
    Action Input: [The input to the action]


    Begin!
    Question: {input}
    Thought: {agent_scratchpad}
    """,
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'])

    #prompt = hub.pull("hwchase17/react")
 
    search_api = TavilySearchResults(max_results=2)
    tools = [answering,create_image]
    llm_with_tool = llm.bind_tools(tools)
    

    agent = create_react_agent(llm=llm,tools=tools,prompt=prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True,memory=memory,handle_parsing_errors=True)
    #image = cv2.imread("/home/dexter/mlops/Langchain/langgraph/tooltool/How-to-use-DallE-3-in-ChatGPT.webp")
    response = agent_executor.invoke({'input':query})
  
    if isinstance(response['output'],str):
    
        return response['output']
    else :
        
        return StreamingResponse(io.BytesIO(response['output'].tobytes()), media_type="image/png")
    
    
