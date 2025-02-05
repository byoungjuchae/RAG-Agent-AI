import httpx
import docker
import asyncio


async def main(query:str):
    url = 'https://localhost:8000/start'
    async with httpx.AsyncClient() as client:
        
        
        timeout = httpx.Timeout(10000.0,connect=5.0)
        import pdb
        pdb.set_trace()
        data= {'query':query}
        response = await client.post(url,data=data,timeout=timeout)
        
        if response.stauts_code == 200:
            
            
            image_data = io.BytesIO(response.content)
            image = Image.open(image_data)
            file_name = os.path.basename()            
        
        
        
if __name__ == '__main__':
    
    
    asyncio.run(main('create the image'))