from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser



url = "https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel"

loader = RecursiveUrlLoader(url=url, max_depth=100, extractor=lambda x : Soup(x,'html.parser').text)
docs = loader.load()

d_sorted = sorted(docs, key=lambda x : x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)
import pdb
pdb.set_trace()

prompt = PromptTemplate(template="""
                        You are an coding assistant with expertise in LCEL, Langchain expresssion language. 
                        Here is a full set of LCEL documentation : \n -------- \n {context} \n ------ \n Answer the 
                        question based on above provided documentation. Ensure any code you provide can be executed \n 
                        with all required imports and variables defined. Structure your answer with a description of the code solution. \n
                        Then list the imports. And finally list the functioning code block. Here is the user question:
                        {question}.
                        """,input_variables=["context","question"])

llm = ChatOllama(model='llama3.1:latest',temperature=0)

chain = prompt | llm | StrOutputParser()
print(chain.invoke({"context":concatenated_content,"question":"How do I build a RAG chain in LCEL?"}))
