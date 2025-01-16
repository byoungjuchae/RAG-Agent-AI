from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.tavily_search import TavilySearchResults 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import torchvision.transforms as T
import onnxruntime as ort
import numpy as np
import cv2
import os
import onnx
import base64
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionPipeline


os.environ['TAVILY_API_KEY'] = ""
providers = ['CUDAExecutionProvider']


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
def sr_model(image: str) -> str:
    """
    This tool is designed for tasks related to Super-Resolution (SR) models, which aim to enhance the resolution and quality of images or videos. 
    It utilizes advanced machine learning techniques, often including deep learning-based architectures like convolutional neural networks (CNNs), generative adversarial networks (GANs), or transformer models. SR models improve image clarity by predicting high-resolution details from low-resolution inputs, making them valuable in fields like medical imaging, video enhancement, and computer vision. 
    Use this tool when you need to upscale images, sharpen details, or apply SR models for real-world applications like satellite imagery, facial recognition, or any domain requiring high-quality visuals from limited data.
    """
    ort_session = ort.InferenceSession('esrgan.onnx',providers=providers)
    transform = T.ToTensor()
    image = cv2.imread(image)
    image= transform(image).unsqueeze(0)
    image = image.numpy()
    inputs = {ort_session.get_inputs()[0].name: image}
    output = ort_session.run(None,inputs)
    output_image = output[0][0].transpose(1,2,0).astype(np.float32)
    output_image = Image.fromarray(output_image)
    image_path= 'SR.png'
    cv2.imwrite('SR.png',output_image)
    
    return f"SR 이미지가 생성되었습니다:{image_path}"
    #_, img_encoded = cv2.imencode('.jpg',output_image)
 
@tool 
def answering(query:str) -> str:
    """
    Use this tool when responding to queries that are not related to coding. 
    It handles a wide range of non-technical questions, including general knowledge, current events, conceptual topics, or creative content. 
    Whether you need information on subjects like science, history, literature, or even more abstract queries, this tool is designed to provide comprehensive and accurate responses. 
    It is particularly useful for exploring real-world applications, detailed explanations, or offering insights into diverse topics beyond coding.
    """
    llm = ChatOllama(model='llama3.1:latest')
    chain = llm | StrOutputParser()
    
    response = chain.invoke(query)
    return response    
@tool
def create_image(query:str) -> str:
    """
     This tool provides the functionality to generate or create images through a diffusion process. 
     Diffusion models are a type of generative model that progressively denoise an input of random noise to produce detailed images that align with user-provided prompts or conditions. 
     This tool can be used for tasks such as image generation, style transfer, and more.
    """
   
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to('cuda')
    image = pipe(query).images[0]
    image_path = 'image.png'
    image.save('image.png',format='JPEG')
    #img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    #img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    
    return  f"이미지가 생성되었습니다: {image_path}"

@tool
def code_execute(query:str) -> str:
    """
    If you have a query that is specific to a programming language and requires more general information, this tool can help. 
    This tool is particularly useful for exploring a wide range of topics, including current events, detailed explanations, creative writing, or research assistance. 
    It bridges the gap between technical and non-technical inquiries, ensuring that you get the most accurate and relevant information for your needs.
    """
    prompt = PromptTemplate.from_template(template="""
                                          You are a excellent programming developer . 
                                          Answer about the {query} in varierty programming language. NO # in the response.
                                          answer will be included only python code. not description. 
                                          """,input_variable=[query])
    llm = ChatOllama(model='llama3.1:latest')
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({'query':query})

    file_name = "example.py"

    with open(file_name, "w") as file:
        
        lines = response.split('\n')
        for line in lines:
            file.write(line+'\n')
    file.close()
        
    print(f"{file_name} has been created successfully.")
    return response
    