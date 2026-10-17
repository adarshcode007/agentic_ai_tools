import os
import shutil
import chromadb

def main():
    db_path = "./chroma_db"

    # Cleasn previous run data for a deterministic fresh start
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    print("=" * 65)
    print("PART 1: INITIALIZE PERSISTENT CHROMADB CLIENT")
    print("=" * 65)

    # PersistentClient automatically initializes SQLite tables and index directories on disk
    client = chromadb.PersistentClient(path=db_path)

    # Create a collection with explicit Cosine similarity metric
    # hnsw:space can be: 'cosine', 'l2', or 'ip'
    collection_name = "security_policies"
    collection = client.create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection '{collection_name}' created with Cosine space.")

    # PART 2: INGEXT DOCUMENTS WITH RICH METADATA
    print("\n" + "=" * 65)
    print("PART 2: INGESTING DOCUMENTS AND METADATA")
    print("=" * 65)

    # Real-world enterprise sample documents
    documents = [
        "All employee laptops must enforce full-disk encryption via BitLocker or FileVault.",
        "Production database credentials must be rotated every 90 days via HashiCorp Vault.",
        "Travel expense reports must be submitted within 30 days of trip completion.",
        "Engineering teams must conduct weekly backlog grooming and sprint planning.",
        "Multi-factor authentication (MFA) is strictly required for all VPN and SSO logins.",
        "Annual financial performance reviews are scheduled for the fourth quarter."
    ]

    metadatas = [
        {"department": "security", "doc_type": "policy", "year": 2024},
        {"department": "infrastructure", "doc_type": "standard", "year": 2023},
        {"department": "finance", "doc_type": "policy", "year": 2024},
        {"department": "engineering", "doc_type": "process", "year": 2023},
        {"department": "security", "doc_type": "policy", "year": 2024},
        {"department": "finance", "doc_type": "review", "year": 2024}
    ]

    # Every item in Chroma MUST have a unique string ID
    ids = [f"doc_{i+1}" for i in range(len(documents))]

    # Ingest: Chroma automatically embeds text using all-MiniLM-L6-v2 under the hood
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Ingested {collection.count()} documents successfully.")

    # PART 3: UNFILTERED SEMANTIC SEARCH

    print("\n" + "=" * 65)
    print("PART 3: UNFILTERED SEMANTIC SEARCH")
    print("=" * 65)

    query_text = "How do we protect user accounts and logins?"
    print(f"Query: '{query_text}'")

    results = collection.query(
        query_texts=[query_text],
        n_results=2
    )

    print("\nTop 2 Unfiltered Matches:")
    for rank in range(len(results["ids"][0])):
        doc_id = results["ids"][0][rank]
        doc_text = results["documents"][0][rank]
        distance = results["distances"][0][rank]
        metadata = results["metadatas"][0][rank]
        print(f"   Rank {rank+1}: ID={doc_id} | Distance={distance:.4f} | Dept={metadata['department']}")
        print(f"        Text: \"{doc_text}\"")
    
    # PART 4: METADATA-FILTERED VECTOR SEARCH
    print("\n" + "=" * 65)
    print("PART 4: FILTERED SEARCH (WHERE dept == 'finance')")
    print("=" * 65)

    # User asks a security/login question, but we filter strictly for Finance docs
    results_filtered = collection.query(
        query_texts=[query_text],
        n_results=2,
        where={"department": "finance"}    # Strict SQL-like pre-filter
    )

    print("Filtered Matches (Strictly department: finance):")
    for rank in range(len(results_filtered["ids"][0])):
        doc_id = results_filtered["ids"][0][rank]
        doc_text = results_filtered["documents"][0][rank]
        distance = results_filtered["distances"][0][rank]
        metadata = results_filtered["metadatas"][0][rank]
        print(f"  Rank {rank+1}: ID={doc_id} | Distance={distance:.4f} | Dept={metadata['department']}")
        print(f"          Text: \"{doc_text}\"")
    
    # PART 5: VERIFYING DISK PERSISTENCE
    print("\n" + "=" * 65)
    print("PART 5: VERIFYING DISK PERSISTENCE")
    print("=" * 65)

    # Deliberately drop the client variable to simulate application shutdown
    del client
    del collection

    # Reconnect a new client instance pointing to the existing folder
    new_client = chromadb.PersistentClient(path=db_path)
    persisted_collection = new_client.get_collection(name=collection_name)

    print(f"Reconnected from disk. Persisted document count: {persisted_collection.count()}")
    assert persisted_collection.count() == 6, "Persistence verification failed!"
    print("Persistence verification SUCCESSFUL!")

if __name__ == "__main__":
    main()
