from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import List, Union
import asyncio
import os

class State(BaseModel):
    message: List[Union[HumanMessage, str]] = Field(default_factory=list)
    generate_message: str = Field(default='This is the result of the generation messages')
    reflect_message: str = Field(default='This is the result of the reflection about the generations.')
    number : int = Field(default=0)
llm = ChatOllama(model='llama3.1:8b')

async def generate(state: State) -> State:
    prompt_generate = ChatPromptTemplate.from_messages(
        [
            ('user', 'You are a essay writer.'),
            MessagesPlaceholder(variable_name='messages')
        ]
    )

    chain_generate = prompt_generate | llm
    state.message += [HumanMessage(content=state.reflect_message)]
    result = await chain_generate.ainvoke({"messages": state.message})

    return State(generate_message=result.content)

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
    return State(reflect_message=result.content,number=state.number)


builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)
builder.set_entry_point('generate')


def should_continue(state: State):

    if state.number > 6:
        return END
    return "reflect"

builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

async def main():
    async for event in graph.astream(
        State(
            message=[
                HumanMessage(content="Generate an essay on the topicality of The Little Prince and its message in modern life")
            ],
            
        ),
        config=config
   
    ):
        print(event)
        print("---")


asyncio.run(main())
