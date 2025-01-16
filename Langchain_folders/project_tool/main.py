from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import Union
from diffusers import FluxPipeline
import os
import torch


os.environ['LANCHAIN_API_KEY'] = ""
os.environ['LANGCHAIN_ENDPOINT'] = ""
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = ""




# @tool
# def rag():
    
    
    
    
    
    
@tool
def create_image(query):
    """
     This tool provides the functionality to generate or create images through a diffusion process. 
     Diffusion models are a type of generative model that progressively denoise an input of random noise to produce detailed images that align with user-provided prompts or conditions. 
     This tool can be used for tasks such as image generation, style transfer, and more.
    """
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",torch_dtype=torch.bfloat16).to("cuda")
    #pipe.enable_model_cpu_offload()
    image = pipe(query,height=1024,width=1024).images[0]
    image_path = 'image.png'
    image.save('image.png',format='JPEG')
    return image_path
    
    
    
# @tool
# def image_assess():
    
    
    
    
    
    
def main(query):
    memory = ConversationBufferWindowMemory(return_messages= True, memory_key='chat_history',input_key='input',k=5)
    llm = ChatOllama(model="llama3.2:latest")
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `create_image` tool for generating the image.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
    tools = [create_image]
    
    agent = create_tool_calling_agent(llm,tools,prompt)
    #agent = create_react_agent(llm=llm,tools=tools,prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=False,memory=memory,handle_parsing_errors=True)
    
    response = agent_executor.invoke({'input':query})
    print(response)    
    
    
if __name__ == '__main__':
    
    main("Generate a description of a ironman image and stop when finished.")