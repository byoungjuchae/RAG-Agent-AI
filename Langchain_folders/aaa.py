import os
import requests
from bs4 import BeautifulSoup
import ollama
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

messages = [
    {'role':'user','content':'Describe some of the business application of Generative AI'}
]
system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."
class Website:
    
    def __init__(self,url):
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }

        self.url = url
        response = requests.get(url,headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script","style","img","input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n",strip=True)
        
def user_prompt_for(website):
    
    user_prompt = f"You arelooking at a website titled {website.title}"
    user_prompt += """\n The contents of this website is as follows \\
            please provide a short summary of this website in markdown. \\
            If it inclues new or announcements, then summarize these too. \n\n
            """
    user_prompt += website.text
    print(website.text)
    return user_prompt
        
# payload = {
#     "model":MODEL,
#     "messages": messages,
#     "stream":False
# }
def message_for(website):
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

response = ollama.chat(model=MODEL, messages=message_for(Website('https://cnn.com')))
# response = requests.post(OLLAMA_API,json=payload,headers=HEADERS)
print(response['message']['content'])
