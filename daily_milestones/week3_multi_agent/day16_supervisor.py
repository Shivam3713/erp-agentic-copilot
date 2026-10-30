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

# 1. Fix the import pathing and load secrets
current_dir = os.path.dirname(os.path.abspath(__file__))

# Map the path to the week2 folder so Python can find Day 14
week2_dir = os.path.abspath(os.path.join(current_dir, "..", "week2_hybrid_rag"))
sys.path.append(week2_dir)

# Load the .env file from the root directory
load_dotenv(os.path.join(current_dir, "..", "..", ".env"))

# Import the existing state and technical nodes from Day 14
from day14_memory import AgentState, retrieve_node, generate_node

# ==========================================
# 2. ROUTER SCHEMA
# ==========================================
class RouteDecision(BaseModel):
    """The strict JSON format the router LLM must follow."""
    route: Literal["technical_support", "general_chitchat"] = Field(
        description="Route to 'technical_support' for MongoDB or database issues. Route to 'general_chitchat' for greetings or general questions."
    )

# ==========================================
# 3. SUPERVISOR & CHITCHAT NODES
# ==========================================
def router_node(state: AgentState) -> str:
    """Classifies the question and decides which node executes next."""
    print("\n--- [NODE: SUPERVISOR ROUTER] ---")
    question = state["question"]
    
    # Initialize the fast 8B model for cheap, instant routing
    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0, 
        api_key=os.getenv("GROQ_API_KEY"),
        http_client=httpx.Client(verify=False)
    )
    
    # Force the LLM to output our Pydantic schema
    structured_router = llm.with_structured_output(RouteDecision)
    
    prompt = f"Analyze this user message: '{question}'. Decide the appropriate route."
    decision = structured_router.invoke(prompt)
    
    print(f"Supervisor Decision: --> {decision.route.upper()}")
    
    # Return the string that dictates the conditional edge mapping
    return decision.route

def chitchat_node(state: AgentState):
    """Handles general greetings without wasting database resources."""
    print("\n--- [NODE: CHITCHAT AGENT] ---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    formatted_history = "\n".join(chat_history) if chat_history else "No previous conversation."
    
    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0.7, 
        api_key=os.getenv("GROQ_API_KEY"),
        http_client=httpx.Client(verify=False)
    )
    
    prompt = f"""You are a helpful Capgemini AI assistant. Keep it brief and friendly.
    Conversation History:
    {formatted_history}
    
    User: {question}
    AI:"""
    
    response = llm.invoke(prompt)
    
    # FIX: Append the greeting to the SQLite checkpointer so it remembers the user's name
    new_history_entries = [f"User: {question}", f"AI: {response.content}"]
    
    return {
        "answer": response.content,
        "chat_history": new_history_entries
    }

# ==========================================
# 4. GRAPH COMPILATION
# ==========================================
def build_rag_graph():
    workflow = StateGraph(AgentState)
    
    # Add all three nodes
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("retrieve", retrieve_node)   
    workflow.add_node("generate", generate_node)   
    
    # The Conditional Edge Logic (Dynamic Routing)
    workflow.add_conditional_edges(
        START,          
        router_node,    
        {
            "general_chitchat": "chitchat",
            "technical_support": "retrieve" 
        }
    )
    
    # Standard edges for the technical pathway
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # The chitchat node ends immediately
    workflow.add_edge("chitchat", END)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(os.path.join(current_dir, "..", "week2_hybrid_rag", "enterprise_memory.sqlite"))
    conn = sqlite3.connect("enterprise_memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)

# ==========================================
# 5. EXECUTION & TESTING
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("DAY 16: MULTI-AGENT SUPERVISOR ROUTING")
    print("="*50)
    
    rag_app = build_rag_graph()
    thread_config = {"configurable": {"thread_id": "routing_test_session_01"}}
    
    # --- TEST 1: The Chitchat Route ---
    print("\n[TEST 1: Sending a casual greeting]")
    user_input_1 = "Hi there! I'm an engineer testing your routing logic today."
    state_1 = rag_app.invoke({"question": user_input_1, "context": [], "answer": ""}, config=thread_config)
    print(f"\nFinal Output 1:\n{state_1['answer']}\n")
    print("-" * 30)
    
    # --- TEST 2: The Technical Route ---
    print("\n[TEST 2: Sending a complex technical issue]")
    user_input_2 = "My MongoDB free tier is throwing an HTTP 504. How do I adjust the maxPoolSize?"
    state_2 = rag_app.invoke({"question": user_input_2}, config=thread_config)
    print(f"\nFinal Output 2:\n{state_2['answer']}\n")