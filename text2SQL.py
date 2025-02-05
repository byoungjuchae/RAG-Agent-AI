from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough






if __name__ == "__main__":
    
    llm = ChatOllama(model='llama3.2:latest')

    prompt_text = "You are an SQL master to translate the {question} to SQL grammar. You only respond the SQL."

    prompt = ChatPromptTemplate.from_template(prompt_text)

    chain = ({"question":RunnablePassthrough()} | prompt | llm | StrOutputParser())

    query = "What movie does have a high rate?"
    print(chain.invoke({'question':query}))