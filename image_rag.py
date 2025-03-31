from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pymilvus import MilvusClient
from glob import glob
import os
import clip
from PIL import Image
import torch
import base64 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)




data_dir = (
    "./images_folder" 
)
image_list = glob(
    os.path.join(data_dir, "images", "*.jpg")
)  


image_dict = {}
print('Image Embedding')
for image_path in image_list:

    embedding= model.encode_image(preprocess(Image.open(image_path)).unsqueeze(0).cuda()).squeeze(0)
    image_dict[image_path] = embedding.detach().cpu().numpy().tolist()   

client = MilvusClient('milvus_demo.db')


client.create_collection(collection_name='image_rag_db',
                        auto_id=True,
                        dimension=512,
                        enable_dynamic_field=True)

print('Insert Dataset')
client.insert(
    collection_name="image_rag_db",
    data=[{"image_path": k, "vector": v} for k, v in image_dict.items()],
)

query = preprocess(Image.open(os.path.join('./images_folder','leopard.jpg'))).unsqueeze(0).cuda()

query_embedding = model.encode_image(query).squeeze(0)

text = clip.tokenize(["a phone case with a leopard on it"]).to(device)

text_embedding = model.encode_text(text).squeeze(0)

query = torch.cat([query_embedding,text_embedding],dim=0)
print('Search data')
search_results = client.search(
    collection_name="image_rag_db",
    data=[text_embedding.detach().cpu().numpy().tolist()],
    output_fields=["image_path"],
    limit=9, 
    search_params={"metric_type": "COSINE"},  
)[0]

retrieved_images = [hit.get("entity").get("image_path") for hit in search_results]

print(retrieved_images[0])
vllm = ChatOllama(model='llama3.2-vision:latest')

with open(retrieved_images[0], "rb") as image_file:
 
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "{question}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]
        }
    ]
prompt = ChatPromptTemplate.from_messages(messages)

chain = {'question':RunnablePassthrough()} | prompt | vllm | StrOutputParser()
response = chain.invoke({'question':"a phone case with a leopard on it"})
print(response)
