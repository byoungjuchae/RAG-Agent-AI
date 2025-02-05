from typing import Annotated, Literal, Sequence, TypedDict
from lagnchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama.chat_models import ChatOllama
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGrpah,START
from langgraph.prebuilt import ToolNode


def grade_documents(state) -> Literal["generate","rewrite"] :
    
    print("---CHECK RELEVANCE---")
    class grade(BaseModel):
        
        binary_score : str = Field(description="Relevance score 'yes' or 'no'")
        
    model = ChatOllama(model='llama3.1:latest',temperature=0)
    llm_with_tool = model.with_structured_output(grade)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    
    chain = prompt | llm_with_tool
    
    messages = state['messages']
    last_message = message[-1]
    
    question = messages[0].content
    docs = last_message.content
    
    scored_result = chain.invoke({"question":question,"context":docs})
    score = scored_result.binary_score       
    
    
def agent(state):
    
    print('---CALL AGENT---')
    messages = state['messages']
    model = ChatOllama(model='llama3.1:latest',temperature=0)
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    
    return {'messages':[response]}


def rewrite(state):
    
    
    print("---TRANSFORM QUERY---")
    messages = state['messages']
    question = messages[0].content
    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]
    model= ChatOllama(model='llama3.1:latest',temperature=0)
    response = model.invoke(response)
    
    return {"messages": [response]}

def generate(state):
    
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    
    question = messages[0].content
    docs = last_message.content
    
    prompt = hub.pull('rlm/rag-prompt')
    llm = ChatOllama(model='llama3.1:latest',temperature=0)
    def format_docs(docs):
        
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context":docs, "question":question})
    return {"messages":[response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent",agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node('retrieve',retrieve)
workflow.add_node("rewrite",rewrite)
workflow.add_node("generate",generate)

workflow.add_edge(START,"agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools":"retrieve",
        END:END,
    }
)
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate",END)
workflow.add_edge("rewrite","agent")
graph = workflow.compile()


