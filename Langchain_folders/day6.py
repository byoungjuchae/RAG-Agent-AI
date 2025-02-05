import os
import requests 
import json 
import ollama
import gradio as gr



os.environ['CUDA_VISIBLE_DEVICES']='0'

def message_llama(prompt):
    messages=[
        {'role': 'system', 'content': 'you are an assistant'},
        {'role':'user','content':prompt}
    ]
    response = ollama.chat(model='llama3.2',messages=messages)
    #response2 = ollama.chat(model='llama3.1',messages=messages)
    
    return response['message']['content']


def stream_llama(prompt):
    messages=[
        {'role': 'system', 'content': 'you are an assistant'},
        {'role':'user','content':prompt}
    ]
    response = ollama.chat(model='llama3.2',messages=messages,stream=True)
    result =""
    for chunk in response:
       
        result += chunk['message']['content']
        yield result   

def stream_llama(prompt,model):
    messages=[
        {'role': 'system', 'content': 'you are an assistant'},
        {'role':'user','content':prompt}
    ]
    response = ollama.chat(model=model,messages=messages,stream=True)
    result =""
    for chunk in response:
       
        result += chunk['message']['content']
        yield result   
#gr.Interface(fn=message_llama,inputs="textbox",outputs="textbox",allow_flagging="never").launch(share=True)

# view = gr.Interface(fn=stream_llama,
#                     inputs=[gr.Textbox(label='Your message:',lines=6)],
#                     outputs=[gr.Markdown(label="Response:")],
#                     allow_flagging="never")

view = gr.Interface(fn=stream_llama,
                    inputs=[gr.Textbox(label='Your message:'), gr.Dropdown(["llama3.1","llama3.2"],label="Select model")],
                    outputs=[gr.Markdown(label='Response:')],
                    allow_flagging="never")

view.launch()