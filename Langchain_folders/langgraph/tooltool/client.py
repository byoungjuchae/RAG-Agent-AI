import httpx 
import asyncio
import io
from PIL import Image



async def start(query:str):
    
    url = 'http://localhost:8000/start'
    
    async with httpx.AsyncClient() as client:
        
        timeout = httpx.Timeout(100000.0,connect=1000.0)
        data = {'query':query}
        response = await client.post(url,data=data,timeout=timeout)
        if response.status_code == 200:
            
    
            image_data = io.BytesIO(response.content)
            image = Image.open(image_data)
            image.save('k.png')       
    
            
            
        


if __name__ == '__main__':
    
    asyncio.run(start(query='Create the picture of ironman.'))