import os
import sys
import json
import uvicorn
import aiosqlite
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the Async Checkpointer and the raw logic from Day 16
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from day16_supervisor import create_uncompiled_workflow

app = FastAPI(title="Streaming AI Co-Pilot API")

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    
    async def event_generator():
        db_path = os.path.abspath(os.path.join(current_dir, "..", "week2_hybrid_rag", "enterprise_memory.sqlite"))
        
        # Open an async connection to the database that closes when the stream finishes
        async with aiosqlite.connect(db_path) as conn:
            memory = AsyncSqliteSaver(conn)
            workflow = create_uncompiled_workflow()
            
            # Compile the graph async for this specific web request
            rag_app = workflow.compile(checkpointer=memory)
            
            thread_config = {"configurable": {"thread_id": request.user_id}}
            initial_state = {"question": request.message, "context": [], "answer": ""}
            
            # Execute the stream!
            async for event in rag_app.astream_events(initial_state, config=thread_config, version="v2"):
                kind = event["event"]
                
                if kind == "on_chain_end" and event.get("name") == "router_node":
                    payload = json.dumps({"type": "status", "content": "Routing request..."})
                    yield f"data: {payload}\n\n"
                    
                elif kind == "on_chain_end" and event.get("name") == "retrieve_node":
                    payload = json.dumps({"type": "status", "content": "Database search complete."})
                    yield f"data: {payload}\n\n"

                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        payload = json.dumps({"type": "token", "content": chunk})
                        yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    print("="*50)
    print("DAY 19: SSE STREAMING SERVER INITIALIZATION")
    print("="*50)
    uvicorn.run(app, host="127.0.0.1", port=8000)