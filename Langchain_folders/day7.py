import os
import gradio as gr
import ollama


system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
                the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
                For example, if the customer says 'I'm looking to buy a hat',\
                you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.' \
                Encourage the customer ot buy hats if they are unsure what to get."

def chat_model(message,history):
    
    
    messages = [{'role':'system','content':system_message}]
    for user_message, assistant_message in history:
        messages.append({'role':'user','content':user_message})
        messages.append({"role":"assistant","content":assistant_message})
    
    messages.append({"role":"user","content":message})
    
    stream = ollama.chat(model='llama3.2',messages=messages,stream=True)
    
    response = ""
    for chunk in stream:
        
        response += chunk['message']['content']
        yield response
        
gr.ChatInterface(fn=chat_model).launch()