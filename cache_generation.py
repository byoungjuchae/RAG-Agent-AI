from langchain.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import json
import os
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"]  = "CRAG"
os.environ["LANGSMITH_TRACING"] = "true"


#with open("question_answer.json") as f:
    #data = json.load(f)

data = [  {
        "question": "What is Reward-Guided Speculative Decoding (RSD) and how does it improve efficiency?",
        "answer": "RSD is a decoding method that balances efficiency and accuracy by integrating computationally lightweight draft models with high-quality target models, reducing the number of required floating point operations (FLOPs)."
      },
      {
        "question": "How does RSD differ from traditional speculative decoding?",
        "answer": "Unlike traditional speculative decoding, RSD incorporates reward signals to prioritize high-value draft outputs, ensuring computational efficiency without compromising accuracy."
      },
      {
        "question": "What are the key benefits of RSD?",
        "answer": "RSD improves inference speed, reduces computational costs, and enhances overall model efficiency while maintaining high-quality outputs."
      },
      {
        "question": "How does RSD optimize the trade-off between computational cost and performance?",
        "answer": "RSD dynamically decides when to invoke the target model based on process rewards, allowing it to retain useful draft outputs while reducing redundant computations."
      },
      {
        "question": "What is the role of the draft model in RSD?",
        "answer": "The draft model generates preliminary outputs that are evaluated using a reward function before being refined by the target model."
      },
      {
        "question": "How does RSD improve performance in long-horizon reasoning tasks?",
        "answer": "By balancing computational cost and accuracy, RSD allows models to better handle complex reasoning tasks, such as math and programming challenges."
      },
      {
        "question": "What theoretical insights support RSD's efficiency?",
        "answer": "RSD employs a threshold-based mixture strategy that optimally balances computational resource use while maintaining high-quality outputs."
      },
      {
        "question": "How does RSD compare to traditional inference methods in terms of efficiency?",
        "answer": "RSD achieves up to 4.4' fewer FLOPs while maintaining competitive accuracy levels compared to traditional inference methods."
      },
      {
        "question": "What benchmarks were used to evaluate RSD's performance?",
        "answer": "RSD was tested on challenging reasoning benchmarks, including Olympiad-level tasks, to demonstrate its computational efficiency and accuracy improvements."
      },
      {
        "question": "What are the limitations of RSD?",
        "answer": "While RSD significantly improves efficiency, its reliance on a reward model may introduce additional complexity in model design and training."
      },
      {
        "question": "How does RSD ensure high-quality outputs while reducing computational overhead?",
        "answer": "It selectively refines draft outputs based on reward evaluations, reducing unnecessary invocations of the target model."
      },
      {
        "question": "What industries could benefit from implementing RSD?",
        "answer": "Industries such as AI-powered customer support, automated coding assistance, and large-scale language modeling can benefit from RSD's efficiency improvements."
      },
      {
        "question": "What challenges exist in integrating RSD into existing LLM architectures?",
        "answer": "Challenges include tuning the reward function effectively and ensuring compatibility with diverse model architectures and use cases."
      },
      {
        "question": "What impact does RSD have on real-time AI applications?",
        "answer": "RSD's efficiency improvements enable faster response times and reduced computational costs, making it ideal for real-time AI applications."
      },
      {
        "question": "How does RSD handle uncertainty in draft model outputs?",
        "answer": "RSD evaluates draft model outputs based on their reward scores, discarding low-reward outputs and refining promising ones."
      },
      {
        "question": "Can RSD be used for other tasks beyond language modeling?",
        "answer": "Yes, RSD's principles can be applied to various sequential decision-making tasks, including reinforcement learning and structured prediction."
      },
      {
        "question": "What are the future research directions for improving RSD?",
        "answer": "Future work may explore adaptive reward functions, improved draft model architectures, and broader applications in AI reasoning tasks."
      },
      {
        "question": "How does RSD align with energy-efficient AI research?",
        "answer": "By reducing the number of required computations, RSD contributes to lower energy consumption, aligning with goals of sustainable AI development."
      },
      {
        "question": "What are the broader implications of RSD for AI efficiency?",
        "answer": "RSD represents a step toward more resource-efficient AI models, balancing computational costs with high-quality output generation."
      } ]
qa_pairs=[] 

for item in data:
    
    qa_text = f'"question": "{item["question"]}", "answer": "{item["answer"]}"'
    qa_pairs.append(qa_text)


retriever = """
    You are an AI assistant.  
    Use the following Q&A pairs as context:\n""" + "\n".join(qa_pairs)



prompt_text = """
    You are an AI assistant. Answer the following question based on the provided context.\n\n
    Question: {question}\n\n
    Context:\n{context}\n\n
    Provide a clear and concise answer.
"""

prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOllama(model='llama3.2:latest',temperature=0)
chain = (prompt | llm | StrOutputParser())
response = chain.invoke({'context':retriever,'question':"What is Reward-Guided Speculative Decoding Algorithm?"})

print(response)
