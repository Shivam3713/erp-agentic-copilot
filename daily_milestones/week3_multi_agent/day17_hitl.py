import os
import sys
import sqlite3
import httpx
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. Fix pathing to import previous nodes
current_dir = os.path.dirname(os.path.abspath(__file__))
week2_dir = os.path.abspath(os.path.join(current_dir, "..", "week2_hybrid_rag"))
sys.path.append(week2_dir)
load_dotenv(os.path.join(current_dir, "..", "..", ".env"))

from day14_memory import AgentState, retrieve_node, generate_node

# ==========================================
# 2. UPGRADED ROUTER SCHEMA (Added 'execute_command')
# ==========================================
class RouteDecision(BaseModel):
    route: Literal["technical_support", "general_chitchat", "execute_command"] = Field(
        description="Route to 'execute_command' ONLY if the user asks to drop, delete, or restart a server/database. Route to 'technical_support' for troubleshooting. Route to 'general_chitchat' for greetings."
    )

# ==========================================
# 3. WORKER NODES
# ==========================================
def router_node(state: AgentState) -> str:
    print("\n--- [NODE: SUPERVISOR ROUTER] ---")
    question = state["question"]
    
    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0, 
        api_key=os.getenv("GROQ_API_KEY"),
        http_client=httpx.Client(verify=False)
    )
    
    structured_router = llm.with_structured_output(RouteDecision)
    decision = structured_router.invoke(f"Analyze this user message: '{question}'. Decide the appropriate route.")
    
    print(f"Supervisor Decision: --> {decision.route.upper()}")
    return decision.route

def chitchat_node(state: AgentState):
    print("\n--- [NODE: CHITCHAT AGENT] ---")
    llm = ChatGroq(
        model="openai/gpt-oss-20b", temperature=0.7, 
        api_key=os.getenv("GROQ_API_KEY"), http_client=httpx.Client(verify=False)
    )
    response = llm.invoke(f"You are a helpful Capgemini AI. Keep it brief. User says: {state['question']}")
    
    chat_history = state.get("chat_history", []) + [f"User: {state['question']}", f"AI: {response.content}"]
    return {"answer": response.content, "chat_history": chat_history}

def execute_command_node(state: AgentState):
    """A dangerous node that modifies infrastructure."""
    print("\n--- [NODE: EXECUTION (DANGER ZONE)] ---")
    print("Executing destructive action on the MongoDB Cluster...")
    
    # In a real app, you would run the actual database drop/restart command here.
    action_result = "SUCCESS: The production database cluster has been restarted."
    
    chat_history = state.get("chat_history", []) + [f"User: {state['question']}", f"AI: {action_result}"]
    return {"answer": action_result, "chat_history": chat_history}

# ==========================================
# 4. GRAPH COMPILATION (WITH BREAKPOINTS)
# ==========================================
def build_rag_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("retrieve", retrieve_node)   
    workflow.add_node("generate", generate_node)
    workflow.add_node("execute", execute_command_node) # Add the new node
    
    workflow.add_conditional_edges(
        START,          
        router_node,    
        {
            "general_chitchat": "chitchat",
            "technical_support": "retrieve",
            "execute_command": "execute" # Wire the new route
        }
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("chitchat", END)
    workflow.add_edge("execute", END)
    
    conn = sqlite3.connect("enterprise_memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # === THE MAGIC (NO-MAGIC) HITL FLAG ===
    # We tell LangGraph to freeze if it ever tries to route to the 'execute' node.
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["execute"] 
    )

# ==========================================
# 5. EXECUTION & HUMAN OVERSIGHT
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("DAY 17: HUMAN-IN-THE-LOOP (HITL) BREAKPOINTS")
    print("="*50)
    
    rag_app = build_rag_graph()
    thread_config = {"configurable": {"thread_id": "security_test_01"}}
    
    print("\n[USER]: Please drop the obsolete MongoDB collections and restart the cluster.")
    initial_state = {"question": "Please drop the obsolete MongoDB collections and restart the cluster.", "context": [], "answer": ""}
    
    # 1. Start the graph. It will route, realize it's dangerous, and immediately pause.
    rag_app.invoke(initial_state, config=thread_config)
    
    # 2. Check the SQLite database to see where the graph is paused
    current_state = rag_app.get_state(thread_config)
    
    if "execute" in current_state.next:
        print("\n" + "!"*50)
        print("🚨 GRAPH EXECUTION SUSPENDED 🚨")
        print("The AI has requested permission to run a destructive infrastructure command.")
        print("Pending Node:", current_state.next)
        print("!"*50)
        
        # 3. The Human Review
        user_approval = input("\nDo you authorize this action? (yes/no): ")
        
        if user_approval.lower() == "yes":
            print("\nAuthorization accepted. Resuming graph from SQLite checkpoint...")
            # Passing 'None' tells the graph to just keep going with the existing state
            final_state = rag_app.invoke(None, config=thread_config)
            print(f"\nFinal AI Output:\n{final_state['answer']}")
        else:
            print("\nAuthorization denied. Aborting operation.")
            # In a real app, we would mutate the state here to tell the AI it was rejected.