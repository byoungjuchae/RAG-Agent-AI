import os
import gradio as gr
import ollama
import json

system_message = "You are a helpful assistant in a airlines. You should try to gently encourage \
                the customer to try items that are on sale."

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}


price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}]
def get_ticket_price(destination_city):
    
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city,"Unknown")

def chat_model(message,history):
    
    
    messages = [{'role':'system','content':system_message}] + history + [{'role':'user','content':message}]
    response = ollama.chat(model="llama3.2",messages=messages,tools=tools)
 

    if response['message']['tool_calls']:
        message = response['message']
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = ollama.chat(model='llama3.2',messages=messages)

    return response['message']['content']
        
def handle_tool_call(message):

    tool_call = message['tool_calls']
    
    arguments = tool_call[0]['function']['arguments']
    city = arguments.get('destination_city')
    price = get_ticket_price(city)

    response = {
        "role":"tool",
        "content" : json.dumps({"destination_city":city,"price":price}),
        "tool_call_id":tool_call[0]['function']['name']
        
    }
    
    return response, city

gr.ChatInterface(fn=chat_model,type='messages').launch()