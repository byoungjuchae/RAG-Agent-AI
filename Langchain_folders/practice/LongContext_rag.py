from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.prompts import format_document

# 기본 문서 프롬프트를 생성합니다. (source, metadata 등을 추가할 수 있습니다)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="{page_content}"
)


def combine_documents(
    docs,  # 문서 목록
    # 문서 프롬프트 (기본값: DEFAULT_DOCUMENT_PROMPT)
    document_prompt=DEFAULT_DOCUMENT_PROMPT,
    document_separator="\n",  # 문서 구분자 (기본값: 두 개의 줄바꿈)
):
    # context 에 입력으로 넣기 위한 문서 병합
    doc_strings = [
        f"[{i}] {format_document(doc, document_prompt)}" for i, doc in enumerate(docs)
    ]  # 각 문서를 주어진 프롬프트로 포맷팅하여 문자열 목록 생성
    return document_separator.join(
        doc_strings
    )  # 포맷팅된 문서 문자열을 구분자로 연결하여 반환

def reorder_documents(docs):
    # 재정렬
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    combined = combine_documents(reordered_docs, document_separator="\n")
    print(combined)
    return combined

loader = PyPDFDirectoryLoader('/')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)

docs = loader.load_and_split(text_splitter)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

doc_func = lambda x: x.page_content
docs = list(map(doc_func, docs))

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                        model_kwargs={'device':'cuda'})

vectordb = FAISS.from_texts(docs,embedding=embedding_model)



retriever = vectordb.as_retriever(search_type="similarity_score_threshold",
                                 search_kwargs={"score_threshold": 0.7,"k":3},)


reorder = LongContextReorder()
reorder = reorder.transform_documents(docs)

prompt = PromptTemplate.from_template(
    """ You are an AI engineer. answer me to refer the {context}.
    
    Question:{question}"""
)

llm = ChatOllama(model='llama3:latest',temperature=0)
chain = {'context':retriever | RunnableLambda(reorder_documents),'question':RunnablePassthrough()} | prompt | llm | StrOutputParser()
print(chain.invoke("what is alpha-clip?"))
