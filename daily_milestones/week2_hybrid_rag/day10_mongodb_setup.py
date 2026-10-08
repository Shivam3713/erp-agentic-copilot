import os
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# 1. Cross-file imports to run the previous pipeline
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from day8_data_ingestion import create_mock_runbook, load_documents, chunk_documents
from day9_vector_embeddings import initialize_embedding_model, generate_embeddings

load_dotenv(os.path.join(current_dir, "..", "..", ".env"))

def connect_to_mongodb():
    """Establishes a connection to the MongoDB Atlas cluster."""
    print("\n--- [1] Connecting to MongoDB Atlas ---")
    uri = os.getenv("MONGO_URI")
    
    if not uri:
        print("ERROR: MONGO_URI not found in .env file.")
        sys.exit(1)
        
    client = MongoClient(uri)
    try:
        # Ping the server to verify connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas!")
        return client
    except ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

def push_to_vector_store(client, chunks, vectors):
    """
    Combines the text chunks and mathematical vectors into BSON documents 
    and inserts them into the database.
    """
    print("\n--- [2] Pushing Data to Vector Store ---")
    
    # Define our target database and collection
    db = client["enterprise_rag"]
    collection = db["knowledge_base"]
    
    # Optional: Clear the collection before inserting to avoid duplicates during testing
    print("Clearing existing test data...")
    collection.delete_many({})
    
    # Package the data into MongoDB's expected dictionary format
    documents_to_insert = []
    for i, chunk in enumerate(chunks):
        doc = {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", "unknown"),
            "embedding": vectors[i] # The 384-dimensional array
        }
        documents_to_insert.append(doc)
    
    # Bulk insert for high performance
    result = collection.insert_many(documents_to_insert)
    print(f"Successfully inserted {len(result.inserted_ids)} vectorized documents into MongoDB.")


if __name__ == "__main__":
    print("="*50)
    print("DAY 10: MONGODB ATLAS PROVISIONING & INSERTION")
    print("="*50)
    
    # Execute Day 8 Pipeline
    file_path = create_mock_runbook()
    raw_docs = load_documents(file_path)
    processed_chunks = chunk_documents(raw_docs)
    
    # Execute Day 9 Pipeline
    embedder = initialize_embedding_model()
    vectors = generate_embeddings(processed_chunks, embedder)
    
    # Execute Day 10 Pipeline
    mongo_client = connect_to_mongodb()
    push_to_vector_store(mongo_client, processed_chunks, vectors)
    
    print("\n[Audit] Data ingestion pipeline complete. The knowledge base is now live in the cloud.")