Install environment 

```bash
pip install -r requirements.txt
```

Start FastAPI and evaluate model 

```bash
uvicorn WebSearch:app --reload
```

Go to 

'''
ip address/docs 
'''

This is the example.

I ask same question "What is the famous American law?" 

It is the picture not using tavily search which is surfing the web tool.  
<img src='./src/notavily.png'>

It is the picutre using tavily search
<img src='./src/tavily.png'>

I check these things using Langsmith. 

