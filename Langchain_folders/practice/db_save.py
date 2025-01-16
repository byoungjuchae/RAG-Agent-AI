from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



loader = PyPDFDirectoryLoader('/')
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=0)
pages = loader.load_and_split(text_splitter)

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-l6-v2',
                                        model_kwargs={'device':'cuda'},
                                        encode_kwargs={'normalize_embeddings':True})

# doc_func = lambda x : x.page_content
# docs = list(map(doc_func,pages))
vectorstore = FAISS.from_documents(pages,
                               embedding= embedding_model,
                               distance_strategy = DistanceStrategy.COSINE)
vectorstore.save_local('./db')