from langchain.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


llm = ChatOllama(model='llama3.1:latest',temperature=0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm = llm,
    memory = memory,
    verbose = True
)

conversation.predict(input="Hi, What is your feeling? An, My name is A.")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
print(memory.buffer)
# conversation.predict(input="What is 1+1?")

# conversation.predict(input="What is my name?")