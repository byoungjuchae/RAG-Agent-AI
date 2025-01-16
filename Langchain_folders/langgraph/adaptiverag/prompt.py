from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools import TavilySearchResults

from pydantic.v1 import BaseModel

def web_search(state):
    search = TavilySearchResults(max_results=2)
    llm = ChatOllama(model='llama3.1:latest',temperature=0)
    llm_with_tool = llm.bind_tools([search])
    prompt = PromptTemplate(template=""" You are an expert all of the fields.So, you can answer the {question} with search the Web.
                            """,input_variables=['question'])
    chain = promt | llm_with_tool | StrOutputParser()
    response = chain.invoke({"question":state['question']})
    
    state['generation'] = response
    return state
    
    
def rewrite(state):
    llm = ChatOllama(model='llama3.1:latest',temperature=0)
    
    prompt = PromptTemplate(template="""
                            You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. Here is the {question}
                            """,input_variables=['question'])
    
    response = prompt | llm | StrOutputParser()
    
    state['generation'] = response
    return state
    
    
def retreiver(state):
    
    prompt = PromptTemplate(template=""" Yoou are a expert about the {query}. you retrieve the {documents} and answer
                            the {query}.
                            """,input_variables=['query','documents'])
    
    llm = ChatOllama(model='llama3.1:latest')
    
    chain = prompt | chain | StrOutputParser()
    response = chain.invoke({'query':state['question'],'documents':state['documents']})
    
    state['generation'] = response
    return state


def query_analysis(state):
    
    
    prompt = PromptTemplate(template=""" You are a router to decide the {query}. If {query} is more related to
    the {context}, You should answer the [RETRIEVE]. But if it does not related to the {context}, you should answer the [WEB].
    The answer's format is json file and you only answer [RETRIEVE] or [WEB]. You don't say anything more longer. 
                 """,input_variables=["query","context"])
    
    
    llm = ChatOllama(model='llama3.1:latest',format='json')
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({'query':state['question'],'context':state['documents']})
    state['generation'] = response
    
    return state


def grade(state):
    
    prompt = PromptTemplate(template=""" You are a writer to upgrade the {generation}. And this is the {query}. You refer to the {query} and {generation}.
                            You can answer to upgrade this {generation}. 
                            """,input_variables=["generation","query"])
    llm = ChatOllama(model='llama3.1:latest')
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({'generation':state['generation'],'query':state['question']})
    
    state['generation'] = response
    return state


def relevant(state):
    
    prompt = PromptTemplate(template= """ You are a decider how much related to the {generation} and {query}. If it is much more relevant between {generation} and {query},
                            you can answer the [YES]. But if it's not related to the {generation} and {query}, you can asnwer the [NO]. 
                            Answering format is the json format.and you don't answer the additional sentences, just answer the json format [YES] or [NO].
                            """,input_variables=["generation","query"])
    
    llm = ChatOllama(model='llama3.1:latest',format='json')
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({'generation':state['generation'],'query':state['question']})
    
    state['generation'] = response
    return state

def hallucination(state):
    
    prompt = PromptTemplate(template= """ You are a decider how much related to the {generation} and {query}. If it is much more relevant between {generation} and {query},
                            you can answer the [YES]. But if it's not related to the {generation} and {query}, you can asnwer the [NO]. 
                            Answering format is the json format.and you don't answer the additional sentences, just answer the json format [YES] or [NO].
                            """,input_variables=["generation","query"])
    llm = ChatOllama(model='llama3.1:latest',format='json')
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({'generation':state['generation'],'query':state['question']})
    state['generation'] = response
    return state

def answer(state):
    
    prompt = PromptTemplate(template= """ You are a decider how much related to the {generation} and {query}. If it is much more relevant between {generation} and {query},
                            you can answer the [YES]. But if it's not related to the {generation} and {query}, you can asnwer the [NO]. 
                            Answering format is the json format.and you don't answer the additional sentences, just answer the json format [YES] or [NO].
                            """,input_variables=["generation","query"])
    llm = ChatOllama(model='llama3.1:latest',format='json')
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({'generation':state['generation'],'query':state['question']})
    state['generation'] = response
    return state
    
    