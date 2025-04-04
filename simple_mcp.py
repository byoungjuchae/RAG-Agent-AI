from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import smithery
import mcp 
from mcp.client.websocket import websocket_client
import asyncio


llm = ChatOllama(model='llama3.2:latest')
async def main():
    async with MultiServerMCPClient(
    {
        "desktop-commander": {
            "command": "npx",
            "args": [
                "-y",
                "@smithery/cli@latest",
                "run",
                "@wonderwhy-er/desktop-commander",
                "--key",
                ""
            
            ],
            "transport": "stdio",  
        },
    }
        
    ) as client:
 
        agent = create_react_agent(llm, client.get_tools())


        weather_response = await agent.ainvoke({"messages": " list up /home directory"})

        print(weather_response['messages'][3].content)
    


asyncio.run(main())
