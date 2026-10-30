import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure pathing allows us to import our previous work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the Supervisor graph (Day 16) to avoid HITL freezing on basic API tests
from day16_supervisor import build_rag_graph

# ==========================================
# 1. PYDANTIC WEB SCHEMAS
# ==========================================
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user session (maps to thread_id).")
    message: str = Field(..., description="The actual question or prompt from the user.")

class ChatResponse(BaseModel):
    status: str
    answer: str

# ==========================================
# 2. FASTAPI INITIALIZATION
# ==========================================
app = FastAPI(
    title="Enterprise AI Co-Pilot API", 
    description="A stateless REST interface wrapping a stateful LangGraph agent.",
    version="1.0"
)

# Initialize the graph globally so it isn't rebuilt on every single web request
print("Booting up LangGraph Supervisor Engine...")
rag_app = build_rag_graph()

# ==========================================
# 3. REST ENDPOINTS
# ==========================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Receives a user message, routes it through the AI, and returns the synthesis."""
    try:
        # The web user_id becomes the SQLite primary key
        thread_config = {"configurable": {"thread_id": request.user_id}}
        initial_state = {"question": request.message, "context": [], "answer": ""}
        
        # Invoke the graph. It will route to chitchat or retrieve automatically.
        final_state = rag_app.invoke(initial_state, config=thread_config)
        
        return ChatResponse(
            status="success",
            answer=final_state["answer"]
        )
    except Exception as e:
        # Standard web error handling to prevent the server from crashing
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple ping to verify the server is alive."""
    return {"status": "active", "database": "connected"}

# ==========================================
# 4. SERVER EXECUTION (UVICORN)
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("DAY 18: FASTAPI SERVER INITIALIZATION")
    print("="*50)
    # Run the ASGI server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)