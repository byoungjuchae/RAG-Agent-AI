from langchain_community.chat_models import ChatOllama
from langchain.agents import AgentExecutor,create_react_agent, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate



if __name__ == '__main__':
    
    llm = ChatOllama(model='llama3.2:latest')
    memory = ConversationBufferWindowMemory(return_messages=True,memory_key='chat_history',input_key='input',k=5)
    prompt = PromptTemplate(template="""
    You are an intelligent assistant with access to the following tools: {tools}. Answer the question accurately. If your response is a base64-encoded image, stop immediately and provide it as the final answer without taking any further actions. Otherwise, use up to two actions and then provide a final answer.

    When responding, strictly follow this format:

    Question: [The question you must answer]
    Thought: [Your thoughts about what to do next]
    Action: [The action to take, one of: {tool_names}]
    Action Input: [The input to the action]


    Begin!
    Question: {input}
    Thought: {agent_scratchpad}
    """,
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'])
    
    tools = []