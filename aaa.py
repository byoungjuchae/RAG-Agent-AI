from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
import os


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "RAG"
os.environ["LANGSMITH_API_KEY"] = "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"
def main():
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])
    question = 'What is EfficientDM?'
    prompt = """Answer the question based only on the following context:
{document} question: {question}
    """
    pro = ChatPromptTemplate.from_template(prompt)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    data = '/home/dexter/mlops/doc/2310.03270v4.pdf'
    loader = PyPDFLoader(data)
    docs = loader.load_and_split(text_splitter)
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',model_kwargs={'device':'cuda'})
    retriever = FAISS.from_documents(docs,embedding=embedding).as_retriever()    
    data = retriever.get_relevant_documents(question)
    llm = ChatOllama(model='llama3.2:latest',temperature=0)
    a = ({"document":retriever | format_docs,"question":RunnablePassthrough()} | pro | llm | StrOutputParser())
    print(a.invoke(question))

if __name__ == "__main__":
    
    main()