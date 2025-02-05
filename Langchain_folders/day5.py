import os
import requests
import json
from typing import List
from bs4 import BeautifulSoup
import ollama

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    
    def __init__(self,url):
        
        self.url = url
        response = requests.get(url,headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body,'html.parser')
        self.title = soup.title.string if soup.title else "Not title found"
        
        if soup.body:
            for irrelevant in soup.body(["script","style","img","input"]):
                irrelevant.decompose()
            
            self.text = soup.body.get_text(separator="\n",strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]
    def get_contents(self):
        
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"
    
ed = Website("https://edwarddonner.com")
print(ed.links)

link_system_prompt = """You are proided with a list of links found on a webpage. \
    You are able to decide which of the links would be most relevant to include in a brochure about the company, \
    such as links to an About page, or a Company page, or Careers/Jobs pages.\ """
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links":[
        {"type":"about page", "url": "https://full.url/goes/here/about"},
        {"type":"careers page": "url": "https://another.full.url/careers"}
    ]
}

"""

def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += """
        please decide which of these are relevant web links for a brochure about the company,
        respond with the full https URL in JSON format. \ Do not include Terms of Service, Privacy, email links.\n """
    user_prompt += "Links (some might be relative links): \n"
    user_prompt += "\n".join(website.links)
    return user_prompt

def get_links(url):
    
    website = Website(url)
    response = openai.chat.completions.create(
        model = MODEL,
        messages = [
            {'role':'system','content':link_system_prompt},
            {"role":"user","content":get_links_user_prompt(website)}
        ],
        response_format = {"type":"json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

huggingface = Website("https://huggingface.co")
huggingface.links

get_links("https://huggingface.co")

def get_all_details(url):
    
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url)
    print("Found links:", links)
    for link in links['link']:
        result += f"\n\n{link['type']}\n"
        result += Website(link['url']).get_contents()
    return result

print(get_all_details('https://huggingface.co'))
system_prompt = """ You are an assistant that analyzes the contents of serveral relevant pages from a company website \
    and creates a short brochure about the the company for prospective customers, investors and recruits. Respond in markdown. \ 
    Include details of company culture, custormers and careers/jobs if you have the information.

"""
def get_brochure_user_prompt(company_name, url):
    
    user_prompt = f"You are looking at a company called: {company_name} \n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += get_all_details(url)
    user_prompt = user_prompt[:5.000]
    return user_prompt

get_brochure_user_prompt("HuggingFace", "https://huggingface.co")

def create_brochure(company_name, url):
    response = openai.chat.completions.create(
        model=MODEL,
        messages = [
            {'role':"system","content":system_prompt},
            {'role':'user','content':get_brochure_user_prompt(company_name,url)}
        ],
    )
    result = response.choices[0].message.content
    display(Markdown(result))
    
create_brochure("HuggingFace","https://huggingface.com")

def stream_brochure(company_name,url):
    stream = openai.chat.completions.create(
        model = MODEL,
        messages = [
            {'role':'system','content': system_prompt},
            {'role':'user','content':get_brochure_user_prompt(company_name,url)}
        ],
        stream=True
    )
    
    response = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        response = response.replace("```","").replace("markdown","")
        update_display(Markdown(response),display_id =display_handle.display_id)

stream_brochure("HuggingFace","https://huggingface.co")

    