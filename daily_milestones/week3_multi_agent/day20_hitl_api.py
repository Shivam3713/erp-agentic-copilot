import os
import sys
import sqlite3
import httpx
import uvicorn
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ==========================================
# 1. PATHING & CROSS-MODULE IMPORTS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
week2_dir = os.path.abspath(os.path.join(current_dir, "..", "week2_hybrid_rag"))
sys.path.append(current_dir)
sys.path.append(week2_dir)
load_dotenv(os.path.join(current_dir, "..", "..", ".env"))

from day14_memory import AgentState, retrieve_node, generate_node
from day17_hitl import chitchat_node, execute_command_node

# ==========================================
# 2. CONTEXT-AWARE ROUTER SCHEMA & NODE
# ==========================================
class RouteDecision(BaseModel):
    destination: Literal["technical_support", "general_chitchat", "execute_command"] = Field(
        description=(
            "Route to 'execute_command' ONLY when the user is explicitly ORDERING or COMMANDING an action "
            "(e.g., 'restart the cluster now', 'drop table users', 'delete collections'). "
            "Do NOT route here if the user is merely asking a question ABOUT a restart, checking status, "
            "or inquiring about past events—route those to 'general_chitchat' or 'technical_support'."
        )
    )

def router_node(state: AgentState) -> str:
    print("\n--- [NODE: SUPERVISOR ROUTER] ---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    history_context = "\n".join(chat_history[-4:]) if chat_history else "No prior context."

    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0, 
        api_key=os.getenv("GROQ_API_KEY"),
        http_client=httpx.Client(verify=False)
    )

    structured_router = llm.with_structured_output(RouteDecision)
    prompt = f"""You are a supervisor deciding workflow routing.
Recent Conversation:
{history_context}

Latest User Input: '{question}'

Decide whether this is an imperative command to modify infrastructure, a technical database question, or general inquiry/status follow-up."""

    decision = structured_router.invoke(prompt)
    print(f"Supervisor Decision: --> {decision.destination.upper()}")
    return decision.destination

# ==========================================
# 3. GRAPH COMPILATION WITH BREAKPOINT
# ==========================================
def build_governed_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("retrieve", retrieve_node)   
    workflow.add_node("generate", generate_node)
    workflow.add_node("execute", execute_command_node)
    
    workflow.add_conditional_edges(
        START,          
        router_node,    
        {
            "general_chitchat": "chitchat",
            "technical_support": "retrieve",
            "execute_command": "execute"
        }
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("chitchat", END)
    workflow.add_edge("execute", END)
    
    db_path = os.path.abspath(os.path.join(week2_dir, "enterprise_memory.sqlite"))
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["execute"]
    )

# ==========================================
# 4. PYDANTIC WEB SCHEMAS
# ==========================================
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="The user's session thread_id.")
    message: str = Field(..., description="The user's prompt or command.")

class ChatResponse(BaseModel):
    status: str = Field(..., description="'completed', 'suspended_for_approval', 'approved_and_executed', or 'rejected_by_admin'")
    thread_id: str
    pending_node: Optional[str] = None
    answer: str

class AdminApprovalRequest(BaseModel):
    user_id: str = Field(..., description="The suspended thread_id to review.")
    approved: bool = Field(..., description="True to execute the command, False to block it.")

# ==========================================
# 5. FASTAPI APPLICATION & ENDPOINTS
# ==========================================
app = FastAPI(
    title="Governed Enterprise AI Co-Pilot (HITL API)",
    version="2.0"
)

rag_app = build_governed_graph()

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Standard user endpoint. Automatically suspends if a destructive command is detected."""
    thread_config = {"configurable": {"thread_id": request.user_id}}
    initial_state = {"question": request.message, "context": [], "answer": ""}
    
    rag_app.invoke(initial_state, config=thread_config)
    current_state = rag_app.get_state(thread_config)
    
    if "execute" in current_state.next:
        return ChatResponse(
            status="suspended_for_approval",
            thread_id=request.user_id,
            pending_node="execute",
            answer="🚨 SECURITY GATE: Destructive infrastructure command detected. Execution is paused awaiting Administrator approval at POST /admin/approve."
        )
    
    return ChatResponse(
        status="completed",
        thread_id=request.user_id,
        pending_node=None,
        answer=current_state.values.get("answer", "")
    )

@app.post("/admin/approve", response_model=ChatResponse)
def admin_approval_endpoint(request: AdminApprovalRequest):
    """Administrator endpoint to authorize or reject a suspended graph thread."""
    thread_config = {"configurable": {"thread_id": request.user_id}}
    current_state = rag_app.get_state(thread_config)
    
    if "execute" not in current_state.next:
        raise HTTPException(
            status_code=400, 
            detail=f"Thread '{request.user_id}' has no pending destructive actions to approve."
        )
    
    if request.approved:
        print(f"\n[ADMIN OVERRIDE]: Approved execution for thread '{request.user_id}'")
        final_state = rag_app.invoke(None, config=thread_config)
        return ChatResponse(
            status="approved_and_executed",
            thread_id=request.user_id,
            pending_node=None,
            answer=final_state["answer"]
        )
    else:
        print(f"\n[ADMIN OVERRIDE]: Rejected execution for thread '{request.user_id}'")
        rejection_msg = "🛑 ACTION BLOCKED: An administrator denied your request to modify production infrastructure."
        
        rag_app.update_state(
            thread_config,
            {
                "answer": rejection_msg,
                "chat_history": [
                    f"User: {current_state.values.get('question')}",
                    f"System Admin: Destructive request rejected by admin. Action was blocked."
                ]
            },
            as_node="execute"
        )
        
        return ChatResponse(
            status="rejected_by_admin",
            thread_id=request.user_id,
            pending_node=None,
            answer=rejection_msg
        )

# ==========================================
# 6. SERVER EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("DAY 20: GOVERNED HITL FASTAPI SERVER")
    print("="*50)
    uvicorn.run(app, host="127.0.0.1", port=8000)