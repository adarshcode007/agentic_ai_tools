import os
import time
import numpy as np 
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

def main():
    # Load environment variables from .env
    load_dotenv()
    api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("Please set a valid PINECONE_API_KEY in your .env file!")

    print("=" * 65)
    print("PART 1: INITIALIZE PINECONE CLIENT & SERVERLESS INDEX")
    print("=" * 65)

    # 1. Initialize modern Pinecone v3+ client
    pc = Pinecone(api_key=api_key)

    index_name = "agentic-rag-demo"
    d = 64 # Vector dimension (using 64 for fast benchmarking; in prod: 1536)

    # Check if index already exists to avoid recreation errors 
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    print(f"Existing index in project: {existing_indexes}")

    if index_name not in existing_indexes:
        print(f"Creating Serverless Index '{index_name}' on AWS us-east-1...")
        pc.create_index(
            name=index_name,
            dimension=d,
            metric="cosine",   # Options: 'cosine', 'euclidean', 'dotproduct'
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"    # Default serverless free-tier region
            )
        )
        # Wait for serverless control plane to transition index state to 'Ready' 
        print("Waiting for index provisioning to complete...")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print(f"Index '{index_name}' is READY!")
    else:
        print(f"Index '{index_name}' already exists. Connecting...")
    
    # 2. Connect to the data plane
    index = pc.Index(index_name)
    print(f"Index host endpoint: {index.describe_index_stats().get('namespaces',{})}")

    # PART 2: MULTI-TENANT ISOLATION VIA NAMESPACES 
    print("\n" + "=" * 65)
    print("PART 2: UPSERTING DATA INTO ISOLATED NAMESPACES")
    print("=" * 65)

    np.random.seed(42)

    # Generate synthetic embeddings
    v1 = np.random.random(d).astype(np.float32).tolist()
    v2 = np.random.random(d).astype(np.float32).tolist()
    v3 = np.random.random(d).astype(np.float32).tolist()

    # Tenant 1: Human Resources (Namespace: 'tenant-hr')
    hr_vectors = [
        {
            "id": "hr_01",
            "values": v1,
            "metadata": {"dept": "hr", "doc_type": "policy", "access_level": "public"}
        },
        {
            "id": "hr_02",
            "values": v2,
            "metadata": {"dept": "hr", "doc_type": "compensation", "access_level": "confidential"}
        }
    ]

    # Tenant 2: Engineering (Namespace: 'tenant-engineering')
    # Note: We intentionally give 'eng_01' the exact same vector values as 'hr_01'
    # to prove that Namespaces strictly isolate query results!
    eng_vectors = [
        {
            "id": "eng_01",
            "values": v1,  # Exact same vector direction as hr_01!
            "metadata": {"dept": "engineering", "doc_type": "architecture", "access_level": "internal"}
        },
        {
            "id": "eng_02",
            "values": v3,
            "metadata": {"dept": "engineering", "doc_type": "policy", "access_level": "internal"}
        }
    ]

    # Upsert with explicit namespace routing
    index.upsert(vectors=hr_vectors, namespace="tenant-hr")
    index.upsert(vectors=eng_vectors, namespace="tenant-engineering")
    print("Upserted vectors into 'tenant-hr' and 'tenant-engineering'.")

    # Give Pinecode cloud indexers a moment to flush buffers
    time.sleep(2)

    # Check stats per namespace
    stats = index.describe_index_stats()
    print("Index stats after upsert:")
    print(f"   Total vector count: {stats.total_vector_count}")
    print(f"   Namespaces: {stats.namespaces}")

    # PART 3: PROVING MULTI-TENANT ISOLATION
    print("\n" + "=" * 65)
    print("PART 3: QUERYING WITH NAMESPACE ISOLATION")
    print("=" * 65)

    # Query using v1 (which matches both hr_01 and eng_01 identical vectors)
    query_vector = v1

    # Query strictly inside HR namespaces
    res_hr = index.query(
        vector=query_vector,
        top_k=2,
        include_metadata=True,
        namespace="tenant-hr"
    )
    print("Results from namespace 'tenant-hr'")
    for match in res_hr.matches:
        print(f"   ID: {match.id:<10} | Score: {match.score:.4f} | Dept: {match.metadata['dept']}")

    # Query strictly inside Engineering namespace
    res_eng = index.query(
        vector=query_vector,
        top_k=2,
        include_metadata=True,
        namespace="tenant-engineering"
    )
    print("\nResults from namespace 'tenant-engineering':")
    for match in res_eng.matches:
        print(f"  ID: {match.id:<10} | Score: {match.score:.4f} | Dept: {match.metadata['dept']}")
        
    # PART 4: METADATA PRE-FILTERING
    print("\n" + "=" * 65)
    print("PART 4: METADATA PRE-FILTERING (doc_type == 'policy')")
    print("=" * 65)

    # In HR, hr_01 is 'policy', hr_02 is 'compensation'.
    # Filter strictly for 'policy'
    filtered_res = index.query(
        vector=query_vector,
        top_k=2,
        include_metadata=True,
        namespace="tenant-hr",
        filter={"doc_type": {"$eq": "policy"}}
    )

    print("Filtered Results in 'tenant-hr' (doc_type == 'policy'):")
    for match in filtered_res.matches:
        print(f"   ID: {match.id:<10} | Score: {match.score:.4f} | Type: {match.metadata['doc_type']}")
    
    print("\n" + "=" * 65)
    print("PINECONE DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()

