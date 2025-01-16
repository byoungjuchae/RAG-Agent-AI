from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser



def retrieve(state,documents):
    
    
    prompt = PromptTemplate.from_template(template="""
                                          You are an expert in all fields. You have to retrieve the {documents}.And answer the
                                          {question}
                                          """,input_variables=['documents','question'])
    
    
    llm = ChatOllama(model='llama3.1:latest',temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({'documents':documents,'question':state['messages']})
    
    return {'messages':response}


def grader(state):
    
    prompt = PromptTemplate.from_template(template="""you are an expert to grade the messages.this is the {question} and 
                                                the other one is {generation}.
                                                You could make better results referring the {generation}. """
                                                ,input_variables=['question','generation'])
    llm = ChatOllama(model='llama3.1:latest',temperature=0)
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({'question':state['messages'],'generation':generation})
    
    return {'messages':response}

def docs_relevant(state):
    
    prompt = PromptTemplate.from_template(template="""
                                          you are an expert all the fileds. you decide which one is better solution.
                                          if you decide to {question} is more relevant the documents, then you could answer
                                          "YES" but it is not relevant, you could answer "NO"
                                          """,input_variables=['question'])
    llm = ChatOllama(model='llama3.1:latest',format='json',temperature=0)
    
    chain = prompt | llm | JsonOutputParser()
    
    response = chain.invoke({"question":state['messages']})
    
    return {'messages':response}

def reduce_hallucination(state):
    
    prompt =  PromptTemplate(template="""
                             You are an expert in all fields. so Take a look around the {document} and check the {question}.
                             And hallucination is occured or not in {generation}. And if the hallucination is occured in 
                             {generation},you give the answer YES. But if it does not occurs hallucination, you give me answer NO in json format.
                             """
                             ,input_variables =['document','question','generation'])
    