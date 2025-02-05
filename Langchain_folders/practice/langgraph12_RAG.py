from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from typing import List
from langchain import hub
from langchain.schema import Document
import os

os.environ["TAVILY_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "agent1"
os.environ["LANGCHAIN_API_KEY"] = ''



urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                        model_kwargs={'device':'cuda'})
vectorstores = FAISS.from_documents(doc_splits,embedding=embedding_model)
retriever = vectorstores.as_retriever()
web_tool = TavilySearchResults(max_results=2)

llm = ChatOllama(model='llama3.1:latest',format='json',temperature=0)
prompt = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore or web search. \n
    Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. \n
    You do not need to be stringent with the keywords in the question related to these topics. \n
    Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
    Return the a JSON with a single key 'datasource' and no premable or explanation. \n
    Question to route: {question}""",
    input_variables=["question"],
)
question = "agent memory"
expert_chain = prompt | llm | JsonOutputParser()

docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOllama(model='llama3.1:latest',temperature=0)

def format_docs(docs):
    
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

question = "agent_memory"
generation = rag_chain.invoke({"context":docs,"question":question})

llm = ChatOllama(model='llama3.1:latest',format='json',temperature=0)
prompt = PromptTemplate( template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],)

hallucination = prompt | llm | JsonOutputParser()

llm = ChatOllama(model='llama3.1:latest',format='json',temperature=0)

prompt = PromptTemplate  template=("""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],

)

answer_grader = prompt | llm | JsonOutputParser()
answer_grader.inovke({'question':question,"generation":generation})

llm = ChatOllama(model='llama3.1:latest',temperature=0)

re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})



class GraphState(TypedDict):
    
    question : str
    generation : str
    documents : List[str]
    
    
    
def retrieve(state):
    
    print("---RETRIEVE---")
    question = state["question"]
    
    retrieve = retriever.get_relevant_documents(question)
    
    return {"documents":documents,"question":question}

def generate(state):
    
    
    
    print("---GENERATE---")
    question = state['question']
    documents = state['documents']
    
    generation = rag_chain.invoke({'question':question,'documents':documents})
    return {'question':question,'documents':documents,'generation':generation}


def grade_documents(state):
    
    print('---CHECK DOCUMENT RELEVANCE TO QUESTION ----')

    question = state['question']
    documents = state['documents']
    
    filterd_docs = []
    
    for d in documents:
        
        score =