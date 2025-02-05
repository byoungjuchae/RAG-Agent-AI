from langchain.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import json
import os
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"]  = "CRAG"
os.environ["LANGSMITH_TRACING"] = "true"


with open("cag_file.json") as f:
    data = json.load(f)
qa_pairs=[] 

for item in data:
    
    qa_text = f'"question": "{item["question"]}", "answer": "{item["answer"]}"'
    qa_pairs.append(qa_text)
    
retriever = """
    You are an AI assistant. You need to create 50 questions and answers. 
    Use the following Q&A pairs as context:\n""" + "\n".join(qa_pairs)



prompt_text = """
    You are an AI assistant. Answer the following question based on the provided context.\n\n
    Question: {question}\n\n
    Context:\n{context}\n\n
    Provide a clear and concise answer.
"""

prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOllama(model='llama3.2:latest',temperature=0)
chain = (prompt | llm | StrOutputParser())
response = chain.invoke({'context':retriever,'question':"What is SOP?"})

print(response)
