from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, MessagesState, HumanMessage
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.tools.tavily_search import TavilySearchResults
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.from_documents import PyPDFLoader
from fastapi import FastAPI, UploadFile, File
import os

os.environ['TAVILY_API_KEY'] = ''
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = ''
os.environ['LANGCHAIN_ENDPOINT'] = 'https://langsmith.com' 

app = FastAPI()

class Agent():
    
    def __init__(self):
        llm = ChatOllama(model='llama3.1:latest',temperature=0)
        self.memory = SqliteSaver.from_conn_string(':memory:')

        self.tools = TavilySearchResults(max_results=2)
        self.llm_with_tool = llm.bind_tools([tools])
        self.prompt = PromptTemplate(template="""You are an answer machine.If somebody requests the answer about question.
                                     you must answer it. And if you don't know the answer feel free to use Tavily search tool.
                                     And you also search the {document}""",
                                     )
    def __call__(self):
        
        graph= self.graph_building()
        graph = self.prompt | graph
        return self.prompt | graph
    
    def graph_building(self):
          
        def chatmodel(state: MessagesState):
            
            return {'messages':[self.llm_with_tool.invoke(state['messsages'])]}

        graph_builder = StateGrpah(MessagesState)
        graph_builder.add_node('chatmodel',chatmodel)
        graph_builder.add_conditional_edges('chatmodel',[self.tools])
        graph_builder.add_node('tools',self.tools)
        graph_builder.add_edge('tools','chatmodel')
        graph_builder.set_entry_point('chatmodel')
        graph = graph_build.compile(checkpointer = self.memory)
        return graph


graph = Agent()
UPLOAD_DIRECTORY = Path(__file__).parent / "uploaded_files"
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
@app.post('/post')
def agent(file : UploadFile= File(...), question: str = Form(...)):
     file_path = UPLOAD_DIRECTORY/file.filename
    with open(file_path,"wb") as buffer:
        
        shutil.copyfileobj(file.file,buffer)
    docs = PyPDFLoader(file_path).load()
    text_splitter =RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/mpnet-base-all-v2',
                                    model_kwargs={'device':'cuda'})
    vectorbase = FAISS.from_documents(docs,embedding=embedding)
    retrieval = vectorbase.as_retriever()
    docs = retrieval.get_relevant_documents(question)
    response =graph.invoke(
        {"messages": [HumanMessage(content=question)],"docs" = docs},
        config={"configurable": {"thread_id": 42}}
    )
    
    return response['messages'][-1].content
    

        
    
    
    