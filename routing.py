from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
prompt = PromptTemplate.from_template(
   """주어진 사용자 질문을 `수학`, `과학`, 또는 `기타` 중 하나로 분류하세요. 한 단어 이상으로 응답하지 마세요.

<question>
{question}
</question>

Classification:"""
)

chain = ({'question':RunnablePassthrough()} |
    prompt
    | ChatOllama(model="llama3.2:latest")
    | StrOutputParser()  
)
print(chain.invoke("염기서열에 대해서 알려줘"))