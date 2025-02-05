import langchain
from langchain import LLMChain
from langgraph import LangGraph, TextSplitter
from langserve import LangServe
def build_rag_chain():
    # Define the retriever
    retriever = langchain.llms.TextRetriever(
        model="text-similarity",
        max_docs=5,
        filter_fn=lambda doc: doc["score"] > 0.5,
    )

    # Define the generator
    generator = langchain.llms.TextGenerator(
        model="text-generation",
        max_length=100,
    )

    # Combine the retriever and generator into a RAG chain
    rag_chain = LLMChain(
        input_type="text",
        output_type="text",
        llm=retriever,
        postprocess_fn=generator,
    )

    return rag_chain

# Example usage:
rag_chain = build_rag_chain()
input_text = "What is the capital of France?"
output_text = rag_chain(input_text)
print(output_text)