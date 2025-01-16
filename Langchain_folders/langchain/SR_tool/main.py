from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from langchain.tools import tool
from langchain.prompts import PromptTemplate
import cv2
import numpy as np
import onnxruntime as ort
import torchvision.transforms as T
import os
import io

providers = ['CUDAExecutionProvider']
# os.environ['LANGCHAIN_API_KEY'] = ""
# os.environ['LANGCHAIN_TRACING_V2'] = "false"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = ""

@tool
def SR(image):
    """ Upscale the image."""
    image= transform(image).unsqueeze(0)
    image = image.numpy()
    inputs = {ort_session.get_inputs()[0].name: image}
    output = ort_session.run(None,inputs)
    output_image = output[0][0].transpose(1,2,0).astype(np.float32) * 255.0
    _, img_encoded = cv2.imencode('.jpg',output_image)
    
    return io.BytesIO(img_encoded.tobytes())

app = FastAPI()
transform = T.ToTensor()
ort_session = ort.InferenceSession('esrgan.onnx',proders=providers)
llm_json = ChatOllama(model='llama3.1:latest',format='json',temperature=0)
llm = ChatOllama(model='llama3.1:latest',temperature=0)
transform = T.ToTensor()

@app.post('/')
def start(file : UploadFile= File(...),query : str =Form(...)):
    file_bytes = file.file.read()
    image = cv2.imdecode(np.frombuffer(file_bytes,np.uint8),cv2.IMREAD_COLOR)
    prompt_json = PromptTemplate(template= """You are a assesing whether {query} want to super-resolution or not. Here is the {query}.
                            Give a binary score 'yes' or 'no'.
                            And provide the binaray score as a JSON with single key 'score' and no premable or explanations.""",input_variables=[query])
    prompt = PromptTemplate(template= """You are a expert all about the field.
                            you answer the {query}.""",input_variables=[query])
    
    chain_json = prompt_json | llm_json | JsonOutputParser() 
    chain = prompt | llm | StrOutputParser()

    response = chain_json.invoke({'query':query})
    print("------response-------")
    print(response)
    if response["score"] == "yes" :
        output = SR(image)    
        print("------SR END-------")
        return StreamingResponse(output,media_type="image/jpeg")
    else:
        response = chain.invoke({"query":query}) 
        return response

