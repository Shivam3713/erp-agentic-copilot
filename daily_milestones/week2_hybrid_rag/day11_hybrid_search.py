import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# 1. Cross-file imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from day9_vector_embeddings import initialize_embedding_model

load_dotenv(os.path.join(current_dir, "..", "..", ".env"))

def perform_vector_search(collection, query_vector, limit=20):
    """Executes a pure semantic vector search in MongoDB."""
    print("Executing Vector Search...")
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": limit
            }
        },
        {
            "$project": {
                "_id": 1,
                "text": 1,
                "source": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    return list(collection.aggregate(pipeline))


def perform_keyword_search(collection, query_string, limit=20):
    """Executes a pure exact-match keyword search (BM25) in MongoDB."""
    print("Executing Keyword Search...")
    pipeline = [
        {
            "$search": {
                "index": "default", # The standard Atlas Search index you created in the UI
                "text": {
                    "query": query_string,
                    "path": "text"
                }
            }
        },
        {
            "$limit": limit
        },
        {
            "$project": {
                "_id": 1,
                "text": 1,
                "source": 1,
                "score": {"$meta": "searchScore"}
            }
        }
    ]
    return list(collection.aggregate(pipeline))


def reciprocal_rank_fusion(vector_results, keyword_results, k=60):
    """
    Merges two result lists using the Reciprocal Rank Fusion algorithm.
    RRF Score = 1 / (k + rank)
    """
    print("\nExecuting Reciprocal Rank Fusion (RRF)...")
    rrf_scores = {}
    
    # 1. Score the Vector Results
    for rank, doc in enumerate(vector_results):
        doc_id = str(doc["_id"])
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"doc": doc, "score": 0.0, "v_rank": rank+1, "k_rank": "N/A"}
        rrf_scores[doc_id]["score"] += 1.0 / (k + (rank + 1))
        
    # 2. Score the Keyword Results
    for rank, doc in enumerate(keyword_results):
        doc_id = str(doc["_id"])
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"doc": doc, "score": 0.0, "v_rank": "N/A", "k_rank": rank+1}
        rrf_scores[doc_id]["score"] += 1.0 / (k + (rank + 1))
        rrf_scores[doc_id]["k_rank"] = rank + 1
        
    # 3. Sort by highest total RRF score
    sorted_fused_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return sorted_fused_results


if __name__ == "__main__":
    print("="*50)
    print("DAY 11: HYBRID RAG AGGREGATION PIPELINE")
    print("="*50)
    
    # Setup
    client = MongoClient(os.getenv("MONGO_URI"))
    collection = client["enterprise_rag"]["knowledge_base"]
    embedder = initialize_embedding_model()
    
    # The Test Query
    user_query = "What happens if endpoint /v2/reporting triggers an alert?"
    print(f"\nUser Query: '{user_query}'")
    
    # 1. Embed the query
    query_vector = embedder.embed_query(user_query)
    
    # 2. Run Parallel Searches
    vector_docs = perform_vector_search(collection, query_vector)
    keyword_docs = perform_keyword_search(collection, user_query)
    
    # 3. Merge with RRF
    fused_results = reciprocal_rank_fusion(vector_docs, keyword_docs)
    
    # 4. Audit Output
    print("\n--- Top 3 Fused Results ---")
    for i, result in enumerate(fused_results[:3]):
        doc = result["doc"]
        print(f"\n[Rank {i+1}] RRF Score: {result['score']:.4f}")
        print(f"Vector Rank: {result['v_rank']} | Keyword Rank: {result['k_rank']}")
        print(f"Source: {doc['source']}")
        print(f"Text Snippet: {doc['text'][:100]}...")