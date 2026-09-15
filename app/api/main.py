import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
load_dotenv() #load the dotenv

#create webapp called app to use below with it's title
app = FastAPI(title = "Copilot Project")

#create client object
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"]) #using our api key to use it further


#shape of our response that we need from the client

class ChatRequest(BaseModel):
    message:str #this says that we expect a key value pair of key would be message and value would be anything but in string format
    #this is user define, i can write it as car and it will still work

#now create the api url 
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    #now we need to write a generator function that instead of returning the whole response after getting completed return word by word from resposne as it gets generated
    async def stream_generator():
        try:
            response_stream = await client.aio.models.generate_content_stream(model='gemini-3.6-flash', contents=request.message)
            async for chunk in response_stream:
                if chunk.text:
                    yield f"data:{chunk.text}\n\n"
            yield f"data:[Done]\n\n"
        except Exception as e:
            error_message = str(e)
            yield f"data:[Error] {error_message}\n\n"
            yield f"data:[Done]\n\n"
    return StreamingResponse(stream_generator(), media_type="text/event-stream")