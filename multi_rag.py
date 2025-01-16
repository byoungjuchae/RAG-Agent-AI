import os
import uuid
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from glob import glob
from PIL import Image
import base64
from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
import nltk
import io
import re
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
fpath = '/home/doc/'
fname = '2310.03270v4.pdf'
def extract_pdf_elements(path, fname):
 
    return partition_pdf(
        filename=os.path.join(fpath,fname),
        extract_images_in_pdf=True, 
        infer_table_structure=True,  
        chunking_strategy="by_title", 
        max_characters=4000, 
        new_after_n_chars=3800,  
        combine_text_under_n_chars=2000,  
        image_output_dir_path=path,  
    )





def categorize_elements(raw_pdf_elements):

    tables = []  
    texts = []  
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element)) 
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))  
    return texts, tables


def summarize(text_elements,table_elements):
    
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatOllama(model='llama3.2:latest')
    summarize_chain = {"element":lambda x : x} | prompt | model | StrOutputParser()
    texts_summaries = []
    text_summaries = summarize_chain.batch(texts,{"max_concurrency":5})
    
    table_summaries = []
    table_summaries = summarize_chain.batch(tables,{"max_concurrency":5})
    
    return text_summaries, table_summaries

def image_text(image_folder):
    prompt_text = """Describe this image. Image: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOllama(model='llama3.2-vision:latest')
    images = glob(os.path.join(image_folder,'*'))
    image_chain = {"element": lambda x : x} | prompt | model | StrOutputParser()
    
    
def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]
    
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

def image_describe(image_paths):
    
    model = ChatOllama(model='llama3.2-vision:latest')
    folder = glob(os.path.join(image_paths,'*.jpg'))
    image_chain = prompt_func | model | StrOutputParser()
    image_b64_list = []
    image_summarize_list = []
    for image_path in folder:
        image = Image.open(image_path)
        base_name = os.path.basename(image_path)
        image_b64 = convert_to_base64(image)

        response = image_chain.invoke({"text":"Describe the picutre","image":image_b64})
        image_b64_list.append(image_b64)
        image_summarize_list.append(response)
    
    return image_b64_list, image_summarize_list

def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """


    store = InMemoryStore()
    id_key = "doc_id"


    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )


    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

  
    if text_summaries:
        add_documents(retriever, text_summaries, texts)

    if table_summaries:
        add_documents(retriever, table_summaries, tables)

    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are financial analyst tasking with providing investment advice.\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide investment advice related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """

    # Multi-modal LLM
    model = ChatOllama(temperature=0, model="llama3.2-vision:latest")

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain



    
if __name__ == '__main__':


    
    raw_pdf_elements = extract_pdf_elements(fpath, fname)
    
    texts, tables = categorize_elements(raw_pdf_elements)
    
    text_summaries, table_summaries = summarize(texts,tables)
    
    image_list, image_summaries = image_describe('/home/figures')
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                       model_kwargs={'device':'cuda'})
    

    vector_store = Chroma(collection_name="mm_rag_cj_blog",embedding_function=embeddings)
        
    retriever_multi_vector_img = create_multi_vector_retriever(
    vector_store,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    image_list,
)
    
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
    query="How much time does BayesDiff require, and how long does QAT and PTQ take?"
    import pdb
    pdb.set_trace()
    response = chain_multimodal_rag.invoke(query)
    print(response)