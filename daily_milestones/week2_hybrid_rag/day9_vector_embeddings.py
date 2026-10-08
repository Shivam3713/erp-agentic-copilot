import os
import sys
from day8_data_ingestion import (
    create_mock_runbook,
    load_documents,
    chunk_documents
)

from langchain_huggingface import HuggingFaceEmbeddings


def initialize_embedding_model():
    """
    Downloads and initializes the open-source BGE embedding model locally.
    normalize_embeddings=True ensures vectors are mapped to a unit sphere, 
    making Cosine Similarity math much faster for the database.
    """
    print("\n--- [1] Initializing HuggingFace BGE Model ---")
    print("Loading 'BAAI/bge-small-en-v1.5' (May take 30 seconds on first run to download...)")
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs = {"device":"cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embedder

def generate_embeddings(chunks, embedder):
    """
    Takes the LangChain Document chunks and converts their text into mathematical vectors.
    """
    print("\n--- [2] Generating Vectors ---")
    
    # Extract just the raw strings from the LangChain Document objects
    text_strings = [chunk.page_content for chunk in chunks]
    
    print(f"Batch embedding {len(text_strings)} chunks...")
    # Generate the vectors in a single batch operation
    vectors = embedder.embed_documents(text_strings)
    return vectors

if __name__ =="__main__":
    print("="*50)
    print("DAY 9: VECTOR EMBEDDINGS (MATHEMATICS OF MEANING)")
    print("="*50)
    
    # --- Execute Day 8 Pipeline ---
    file_path = create_mock_runbook()
    raw_docs = load_documents(file_path)
    processed_chunks = chunk_documents(raw_docs)
    # --- Execute Day 9 Pipeline ---
    embedding_model = initialize_embedding_model()
    document_vectors = generate_embeddings(processed_chunks, embedding_model)
    
     # --- Audit Trail ---
    print("\n--- [3] Audit Trail: Inspecting the Math ---")
    
    # Look at the first chunk's vector
    first_vector = document_vectors[0]
    dimension_count = len(first_vector)
    
    print(f"Number of chunks embedded: {len(document_vectors)}")
    print(f"Model Dimensions: {dimension_count} (You will need this exact number for Day 10 MongoDB Setup)")
    
    # Print the first 5 numbers of the 384-length array to prove it's a vector
    print(f"\nMathematical representation of Chunk 1 (First 5 dimensions):")
    print(f"[{first_vector[0]:.5f}, {first_vector[1]:.5f}, {first_vector[2]:.5f}, {first_vector[3]:.5f}, {first_vector[4]:.5f}, ...]")
    
    # Test a single query embedding (simulating a user asking a question)
    user_query = "How do I fix a MongoDB timeout?"
    print(f"\nSimulating User Query: '{user_query}'")
    query_vector = embedding_model.embed_query(user_query)
    print(f"Query embedded successfully into a {len(query_vector)}-dimensional array.")
    
    


    