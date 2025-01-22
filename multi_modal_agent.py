from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent,initialize_agent,AgentType
from langchain_core.output_parsers import StrOutputParser
from diffusers import FluxPipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage
import torch
from io import BytesIO
from PIL import Image
import base64
def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG") 
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def prompt_func(data):
   
    image = data['image']
    text = data['text']

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


@tool
def retriever(query: str) -> str:
    """
    When the input is relevant to the pdf file, you can use this tool. And return the answer
    """
    document = '/home/dexter/mlops/Langchain/langgraph/tooltool/uploaded_files/DemoFusion_ Democratisting High-Resolution Image Genration With No money.pdf'
    docs = PyPDFLoader(document).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                      model_kwargs={'device':'cuda'})
    vectorstore = FAISS.from_documents(docs,embedding=embedding)
    retriever = vectorstore.as_retriever()
    
    qe = retriever.get_relevant_documents(query)
    
    return qe
@tool 
def make_image(prompt : str):
    "When llm wants to make a image, use this tool."
    pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell',torch_dtype=torch.bfloat16).to('cuda')
    
    image = pipe(prompt,height=1024,width=512).images[0]
    base64_image = convert_to_base64(image)
    image_name = './flux-schnell.png'
    image.save('flux-schnell.png')
    # llm_vision = ChatOllama(model='llama3.2-vision:latest',temperature=0)
    # chain = prompt_func | llm_vision | StrOutputParser()
    # response = chain.invoke({'text':'Describe the picture','image':base64_image})
 
    return f"Here is the image url:{image_name}"
     
@tool 
def describe_image(data):
    "When you want to describe the image, use this tool."
    image = Image.open(data)
    data = convert_to_base64(image)
    llm_vision = ChatOllama(model='llama3.2-vision:latest',temperature=0)
    chain = prompt_func | llm_vision | StrOutputParser()

    response = chain.invoke({'text':'Describe the image.','image':data})

    
    # promt = """ summarize the sentence"""
    # prompt = PromptTemplate(template=promt)
    
    # chain = prompt | ChatOllama(model="llama3.1:latest") | StrOutputParser()
    # response = chain.invoke(response)
    return response
    
    


def main(query:str):
    memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=5)
    prompt_temp = """ 
    You are an intelligent assistant with access to the following tools: {tools}.
    You must answer the question. 
    you follow the steps.
    Question: [The question you must answer]
    Thought: [Your thoughts about what to do next]
    Action: [The action to take, one of: {tool_names}]
    Action Input: [The input to the action]


    Begin!
    Question: {input}
    Thought: {agent_scratchpad}
    """
    prompt = PromptTemplate(template= prompt_temp,
                            input_variables=['input','tools','agent_scratchpad']
        
    )
    llm = ChatOllama(model='llama3.2:latest',temperature=0)
    tool = [make_image,describe_image]
    llm_with_tool = llm.bind_tools(tool)

    agent = initialize_agent(llm=llm,tools=tool,prompt=prompt,memory=memory,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,max_iterations=2,verbose=True)
    #agent_executor = AgentExecutor(agent=agent,tools=tool,memory=memory,verbose=True)
    response = agent.invoke({'input':query})
        
        

if __name__ == '__main__':
    query = "Describe the image. follow these steps.First create the person who eats a hamburger. Second describe the image."
    main(query)