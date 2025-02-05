from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser






if __name__ == "__main__":
    
    llm = ChatOllama(model='llama3.2:latest')

    prompt_text = "You are an assistant chatbot that creates SQL queries."

    prompt = PromptTemplate.from_template( prompt_text)

    chain = prompt | llm | StrOutputParser()

    query = "Generate a SQL query to fetch all customers whose age is greater than 30."
    print(chain.invoke(query))