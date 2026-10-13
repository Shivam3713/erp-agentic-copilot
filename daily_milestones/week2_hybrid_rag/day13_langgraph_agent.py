import os
import sys
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import httpx
from pymongo import MongoClient
from dotenv import load_dotenv


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from day9_vector_embeddings import initialize_embedding_model
from day11_hybrid_search import perform_keyword_search, perform_vector_search, reciprocal_rank_fusion
from day12_reranker import initialize_reranker, rerank_documents

load_dotenv(os.path.join(current_dir, "..","..",".env"))


# ==========================================
# 1. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    """The strict dictionary schema that flows through the graph."""
    question: str
    context: List[dict]
    answer: str

# ==========================================
# 2. WORKER NODES
# ==========================================
def rertieve_node(state: AgentState):
    """Executes the Hybrid RAG + Cross-Encoder pipeline."""
    print("\n--- [NODE: RETRIEVAL] ---")
    question = state["question"]
    
    
    #initialize infrastructure
    client = MongoClient(os.getenv("MONGO_URI"))
    collection= client["enterprise_rag"]["knowledge_base"]
    embedder = initialize_embedding_model()
    reranker = initialize_reranker()
    
    print(f"Extracting context for: '{question}'")
    # Execute Pipeline
    query_vector = embedder.embed_query(question)
    vector_docs = perform_vector_search(collection, query_vector, limit = 10)
    keyword_docs = perform_keyword_search(collection, question, limit = 10)
    
    fused_results = reciprocal_rank_fusion(vector_docs, keyword_docs)
    final_results = rerank_documents(question, fused_results, reranker, 3)
    
    # Return partial dict to mutate the 'context' key
    return {"context": final_results}

def generate_node(state: AgentState):
    """Passes context and question to the local LLM to generate an answer."""
    print("\n--- [NODE: GENERATION] ---")
    question = state["question"]
    context_chunks = state["context"]
    
    # Format context for the LLM prompt
    formatted_context = "\n\n".join([res["doc"]["text"] for res in context_chunks])
    
    # Initialize Local LLM
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("CRITICAL: GROQ_API_KEY is missing from the environment.")
        
    # 2. Bypass corporate SSL inspection
    custom_http_client = httpx.Client(verify=False)
        
    # 3. Initialize Cloud LLM via Groq API
    print("Routing LLM synthesis through Groq API (llama3-8b)...")
    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0, 
        api_key=groq_key,
        http_client=custom_http_client
    )
     
    prompt = f"""You are an enterprise technical support AI. Use ONLY the following context to answer the user's question. 
    If the context contains conflicting information, use your technical reasoning to determine the exact fix for the specific error mentioned.
    
    Context:
    {formatted_context}
    
    Question: {question}
    
    Answer:"""
    
    # Execute Synthesis
    response = llm.invoke(prompt)
    return {"answer": response.content}


def build_rag_graph():
    """Assembles nodes and edges into an executable AI application."""
    workflow = StateGraph(AgentState)
    
    # Register the nodes
    workflow.add_node("retrieve", rertieve_node)
    workflow.add_node("generate", generate_node)
    
    # Define the routing logic
    workflow.add_edge(START,"retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("DAY 13: LANGGRAPH RAG SYNTHESIS")
    print("="*50)
    
    rag_app = build_rag_graph()
    
    # The tricky Day 12 question
    user_input = "My MongoDB Atlas free tier is failing with an HTTP 504 error. How do I fix it?"
    
    # Inject initial payload at START
    initial_state = {"question": user_input, "context": [], "answer": ""}
    
    print("\nInvoking LangGraph State Machine...")
    final_state = rag_app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("FINAL LLM SYNTHESIS:")
    print("="*50)
    print(final_state["answer"])


