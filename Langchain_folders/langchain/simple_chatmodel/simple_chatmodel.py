from fastapi import FastAPI
from langchain_ollama import ChatOllama
from pydantic import BaseModel
app = FastAPI()

llm = ChatOllama(model='llama3.1:latest',temperature=0)


class Output(BaseModel):
    
    output : str
    
@app.post('/text')
async def answer(question:str ):
    return Output(output=llm.invoke(question).content)
    
    

    
    