import os
import sys
from sentence_transformers import CrossEncoder
from pymongo import MongoClient
from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from day9_vector_embeddings import initialize_embedding_model
from day11_hybrid_search import perform_vector_search, perform_keyword_search, reciprocal_rank_fusion

load_dotenv(os.path.join(current_dir, "..", "..", ".env"))

def initialize_reranker():
    """Loads the local HuggingFace Cross-Encoder model."""
    print("Loading Cross-Encoder model (ms-marco-MiniLM-L-6-v2)...")
    # This runs locally on CPU, just like the BGE model
    # return CrossEncoder("corss-encoder/ms-macro-MiniLM-L-6-v2", max_length=512)
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)



def rerank_documents(query, fused_results,reranker_model,top_k= 3):
    """Takes the Top N results from the Hybrid Search and scores them 
        by reading the Query and Document side-by-side.
    """
    print(f"\nExecuting Cross-Encoder Re-ranking on top {len(fused_results)} candidates...")
    
    # 1. Prepare the pairs: [[Query, Doc1], [Query, Doc2], ...]
    pairs = []
    for  result in fused_results:
        doc_text = result["doc"]["text"]
        pairs.append([query, doc_text])
    
    scores = reranker_model.predict(pairs)
    
    for i, result in enumerate(fused_results):
        result["cross_encoder_score"] = float(scores[i])
        
    reranker_results = sorted(fused_results, key = lambda x:x["cross_encoder_score"], reverse=True)   
    
    return reranker_results[:top_k]


if __name__ == "__main__":
    print("="*50)
    print("DAY 12: CROSS-ENCODER RE-RANKING PIPELINE")
    print("="*50)
    
    # Setup infrastructure
    client = MongoClient(os.getenv("MONGO_URI"))
    collection  = client["enterprise_rag"]["knowledge_base"]
    
    #initialize both ai model locally
    embedder = initialize_embedding_model()
    reranker = initialize_reranker()
    
    # We ask a tricky question meant to confuse the database
    user_query = "My MongoDB Atlas free tier is failing with an HTTP 504 error. How do I fix it?"
    print(f"\nUser Query: '{user_query}'")
    
    # --- STAGE 1: HYBRID RETRIEVAL (Broad Catch) ---
    query_vector = embedder.embed_query(user_query)
    vector_docs = perform_vector_search(collection, query_vector, limit = 10)
    keyword_docs = perform_keyword_search(collection, user_query, limit = 10)
    
    # Merge the broad catch (Gets the Top 10 combined)
    fused_results = reciprocal_rank_fusion(vector_docs, keyword_docs)
    
    # --- STAGE 2: CROSS-ENCODER (Precision Strike) ---
    final_results =rerank_documents(user_query, fused_results, reranker,3)
    print("\n--- Final Re-Ranked Top 3 Results ---")
    for i, result in enumerate(final_results):
        doc = result["doc"]
        print(f"\n[Rank {i+1}] Cross-Encoder Score: {result['cross_encoder_score']:.4f} (Old RRF: {result['score']:.4f})")
        print(f"Text Snippet: {doc['text'][:120]}...")
    
    
    
    