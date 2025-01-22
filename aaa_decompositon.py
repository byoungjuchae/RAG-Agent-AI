from langchain.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import os

template = """ You are helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):
"""

prompt_decomposition = ChatPromptTemplate.from_template(template)
llm = ChatOllama(model='llama3.1:latest',temperature=0)

de_chain = (prompt_decomposition | llm | StrOutputParser() | (lambda x : x.split("\n")))

question = "What is task decomposition for LLM agents?"

questions = de_chain.invoke({"question":question})

template = """ Here is the question you need to answer:

\n ---- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n
 




"""