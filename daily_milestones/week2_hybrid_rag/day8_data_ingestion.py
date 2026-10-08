import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_mock_runbook():
    """Generates a mock enterprise text file for ingestion."""
    runbook_content = """
    # Capgemini COREai - Infrastructure Runbook
    
    ## Section 1: MongoDB Atlas Outages
    If a MongoDB Atlas cluster (like prod-db-01) experiences an HTTP 504 Gateway Timeout, 
    it indicates connection pool exhaustion. 
    Resolution: Navigate to the Atlas UI, select the cluster, and increase the max pool size to 10,000.
    
    ## Section 2: API Gateway Rate Limiting
    If an unauthorized access alert is triggered on endpoint /v2/reporting, the SOC must 
    immediately quarantine the endpoint using the Cloudflare WAF panel.
    Resolution: Revoke all active bearer tokens issued in the last 24 hours and page the on-call security engineer.
    
    ## Section 3: Billing Gateway Failures
    When the Stripe API returns a 402 Payment Required error, the customer's enterprise tier 
    will automatically downgrade to free after a 3-day grace period.
    """
    
    save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_documents")
    os.makedirs(save_dir, exist_ok=True)
    
    filename = os.path.join(save_dir, "enterprise_runbook.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(runbook_content)
    return filename


def load_documents(file_path: str):
    """Loads a raw file into a LangChain Document object."""
    print(f"\n--- [1] Loading Document: {file_path} ---")
    loader = TextLoader(file_path)
    docs = loader.load()
    
    print(f"Successfully loaded {len(docs)} document(s).")
    # print(docs)
    print(f"Total Character Count: {len(docs[0].page_content)}")
    print(f"Attached Metadata: {docs[0].metadata}")
    
    return docs


def chunk_documents(documents):
    """Splits large documents into overlapping chunks."""
    print("\n--- [2] Semantic Chunking ---")
    
    # We use small sizes here specifically to force multiple chunks for demonstration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} distinct chunks.\n")
    return chunks
    
if __name__ =="__main__":
    print("="*50)
    print("DAY 8: DATA INGESTION & CHUNKING")
    print("="*50)
    
    # 1. Create the mock file
    file_path = create_mock_runbook()
    
    # 2. Load it into memory
    raw_docs = load_documents(file_path)
    
    # 3. Split it into chunks
    processed_chunks = chunk_documents(raw_docs)
    
    # 4. Audit the Output
    print("--- [3] Audit Trail: Inspecting Chunks ---")
    for idx, chunk in enumerate(processed_chunks):
        print(f"\n[Chunk {idx + 1}]")
        print(f"Content length: {len(chunk.page_content)} characters")
        print(f"Text: {chunk.page_content}")
        print(f"Metadata attached: {chunk.metadata}")