from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import psycopg2 as pg2





if __name__ == "__main__":
    
    llm = ChatOllama(model='llama3.2:latest')

    prompt_text = """ You are an expert SQL assistant. Your task is to translate the question into precise, efficient, and accurate SQL queries.  
                    When translating, ensure the following:
                    - Clearly specify all fields, conditions, and sorting criteria required.
                    - Optimize the query for readability and efficiency.
                    - Add explanatory comments to clarify complex logic or conditions.
                    - Confirm the SQL query accurately answers the user's intent.
                    
                    There is a NVDA table having columns (date, high, low, close, volume) related to the stock price. 
        
                    Here is the question:
                    {question}
                    
                    You only respond the SQL.
                """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    chain = ({"question":RunnablePassthrough()} | prompt | llm | StrOutputParser())

    query = "When is the highest stock price at close?"
    
    response = chain.invoke({'question':query})
    
    conn = pg2.connect(host='postgres',
                   dbname ='postgres',
                   user='postgres',
                   password='postgres',
                   port=5432)

    cur = conn.cursor()
    
    cur.execute(f"{response}".format(response))
    row = cur.fetchone()

    print(row)