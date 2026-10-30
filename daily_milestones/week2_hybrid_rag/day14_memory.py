import os
import sys
import sqlite3
import operator
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
import httpx
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

# ==========================================
# 0. CROSS-FILE IMPORTS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from day9_vector_embeddings import initialize_embedding_model
from day11_hybrid_search import perform_keyword_search, perform_vector_search, reciprocal_rank_fusion
from day12_reranker import initialize_reranker, rerank_documents

load_dotenv(os.path.join(current_dir, "..", "..", ".env"))

# ==========================================
# 1. UPGRADED STATE DEFINITION (The Memory Schema)
# ==========================================
class AgentState(TypedDict):
    question: str
    context: List[dict]
    answer: str
    # Annotated + operator.add tells LangGraph: "Append to this list, do NOT overwrite it."
    chat_history: Annotated[List[str], operator.add] 

# ==========================================
# 2. WORKER NODES
# ==========================================
def retrieve_node(state: AgentState):
    print("\n--- [NODE: RETRIEVAL] ---")
    question = state["question"]
    
    client = MongoClient(os.getenv("MONGO_URI"))
    collection = client["enterprise_rag"]["knowledge_base"]
    embedder = initialize_embedding_model()
    reranker = initialize_reranker()
    
    print(f"Extracting context for: '{question}'")
    query_vector = embedder.embed_query(question)
    vector_docs = perform_vector_search(collection, query_vector, limit=10)
    keyword_docs = perform_keyword_search(collection, question, limit=10)
    
    fused_results = reciprocal_rank_fusion(vector_docs, keyword_docs)
    final_results = rerank_documents(question, fused_results, reranker, top_k=3)
    
    # === NEW: THE SANITIZATION STEP ===
    # Iterate through the final results and cast MongoDB ObjectIds to standard strings
    for res in final_results:
        if "_id" in res["doc"]:
            res["doc"]["_id"] = str(res["doc"]["_id"])
    # ==================================
    
    return {"context": final_results}

def generate_node(state: AgentState, config:RunnableConfig):
    print("\n--- [NODE: GENERATION] ---")
    question = state["question"]
    context_chunks = state["context"]
    chat_history = state.get("chat_history", [])
    
    formatted_context = "\n\n".join([res["doc"]["text"] for res in context_chunks])
    
    # Format the history so the LLM knows what was previously discussed
    formatted_history = "\n".join(chat_history) if chat_history else "No previous conversation."
    
    groq_key = os.getenv("GROQ_API_KEY")
    custom_http_client = httpx.Client(verify=False)
        
    print("Routing LLM synthesis through Groq API (gpt-oss-20b)...")
    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0, 
        api_key=groq_key,
        http_client=custom_http_client
    )
     
    prompt = f"""You are an enterprise technical support AI. Use the provided context to answer the question.
    Consider the conversation history if the user refers to previous steps.
    
    Conversation History:
    {formatted_history}
    
    Retrieved Context:
    {formatted_context}
    
    Current Question: {question}
    
    Answer:"""
    
    response = llm.invoke(prompt, config = config)
    
    # We return the answer, PLUS we append both the user's question and the AI's answer to the history list
    new_history_entries = [f"User: {question}", f"AI: {response.content}"]
    
    return {
        "answer": response.content,
        "chat_history": new_history_entries
    }

# ==========================================
# 3. GRAPH COMPILATION WITH SQLITE
# ==========================================
def build_rag_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Establish SQLite connection (creates enterprise_memory.sqlite in your folder)
    conn = sqlite3.connect("enterprise_memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # Compile the graph and bind the checkpointer
    return workflow.compile(checkpointer=memory)

# ==========================================
# 4. MAIN EXECUTION (Testing Memory)
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("DAY 14: LANGGRAPH PERSISTENT MEMORY")
    print("="*50)
    
    rag_app = build_rag_graph()
    
    # Define a strict thread ID to track this specific conversation
    thread_config = {"configurable": {"thread_id": "shivam_session_01"}}
    
    # --- TURN 1 ---
    print("\n[TURN 1: Initial Question]")
    user_input_1 = "My MongoDB Atlas free tier is failing with an HTTP 504 error. How do I fix it?"
    
    # We do NOT pass history in the initial state; LangGraph loads it from SQLite if it exists
    initial_state_1 = {"question": user_input_1, "context": [], "answer": ""}
    state_1 = rag_app.invoke(initial_state_1, config=thread_config)
    print(f"\nAI Response 1:\n{state_1['answer']}\n")
    print("-" * 30)
    
    # --- TURN 2 ---
    print("\n[TURN 2: The Follow-Up (Testing Amnesia)]")
    # Notice we ask a highly ambiguous question. Without memory, the AI would fail here.
    user_input_2 = "Can you explain step 2 from your previous response in more detail?"
    
    # We only pass the new question. LangGraph automatically fetches the context and history from SQLite!
    initial_state_2 = {"question": user_input_2, "context": [], "answer": ""}
    state_2 = rag_app.invoke(initial_state_2, config=thread_config)
    
    print(f"\nAI Response 2:\n{state_2['answer']}\n")