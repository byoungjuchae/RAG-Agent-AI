from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Union
from PIL import Image
import os
import asyncio
import base64
from io import BytesIO


vllm = ChatOllama(model='llama3.2-vision:latest')
llm = ChatOllama(model='llama3.2:latest')


class State(BaseModel):
    
    image : str 
    message : List[Union[HumanMessage, str]] =Field(default_factory=list) 
    generate_message: str =  Field(default='This is the result of the generation messages')
    reflect_message : str = Field(default='This is the result of the reflection about the generations.')
    number : int = Field(default=0)
    
def convert_to_base64(pil_image):

    buffered = BytesIO()
    pil_image.save(buffered,format="JPEG") 
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def prompt_func(data):
   
    image = data['image']
    text = data['text']

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

async def generate(state: State) -> State:

    
    prompt_generate = ChatPromptTemplate.from_messages(
        [
            ('user', 'You are writer for essay.'),
            MessagesPlaceholder(variable_name='messages'),
        ]
    )
    chain_generate = prompt_generate | vllm
    
    image_part = {'type':'image_url','image_url': f"data:image/jpeg;base64,{state.image}"}
    text_part = {'type':'text','text':'Write the essay inspiring this photo'}
    if state.reflect_message != 'This is the result of the reflection about the generations.':
        text_part['text'] += ' and reflect the message.\n\n' + 'Here is the reflect message:\n'+ state.reflect_message

    content_parts = []
    content_parts.append(image_part)
    content_parts.append(text_part)

    result = await chain_generate.ainvoke({'messages':[HumanMessage(content=content_parts)]})
    
    return State(generate_message=result.content,image=state.image)
  
# async def reflect(state: State) -> State:
#     prompt_reflect = ChatPromptTemplate.from_messages(
#         [
#             ('user', 'You are advisor for the essay. You refer the image and advise the essay.'),
#             MessagesPlaceholder(variable_name='messages'),
#         ]
#     )
#     chain_reflect = prompt_reflect | vllm
    
    
#     image_part = {'type':'image_url','image_url': f"data:image/jpeg;base64,{state.image}"}
#     text_part = {'type':'text','text':'Here is the essay:\n\n' + state.generate_message}
    
    
#     content_parts = []
#     content_parts.append(image_part)
#     content_parts.append(text_part)

#     result = await chain_reflect.ainvoke({'messages':[HumanMessage(content=content_parts)]})
#     state.number += 1
#     return State(reflect_message=result.content,number=state.number,image=state.image)
async def reflect(state: State) -> State:

    prompt_reflect = ChatPromptTemplate.from_template(
        """
        You are a revisor for the essay. You check the context of each paragraph.

        Here is the essay:
        {input}
        """
    )
    
    chain_reflect = prompt_reflect | llm
    result = await chain_reflect.ainvoke({'input': state.generate_message})
    state.number +=1
    return State(reflect_message=result.content,number=state.number,image=state.image)
    
def should_end(state):
    
    if state.number > 6:
        return END
    return "reflect"
    
builder = StateGraph(State)
builder.add_node("generate",generate)
builder.add_node('reflect',reflect)
builder.add_edge("reflect","generate")
builder.add_conditional_edges("generate",should_end)
builder.set_entry_point("generate")
memory = MemorySaver()

graph = builder.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "1"}}

async def main():
    image = Image.open('./guardians of galaxy.jpg').convert('RGB')
    data = convert_to_base64(image)
    async for event in graph.astream(
        State(
            message=[
                HumanMessage(content="Generate an essay on the topicality of The Little Prince and its message in modern life")
            ],
            image = data

        ),
        config=config
   
    ):
        print(event)
        print("---")


asyncio.run(main())
