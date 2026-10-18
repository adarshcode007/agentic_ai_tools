import os
import time
import numpy as np 
from dotenv import load_dotenv 
from pinecone import Pinecone, ServerlessSpec 

def build_sparse_vector(text: str, vocab: dict) -> dict:
    """
    Simple sparse encoder: computes Term Frequency (TF) for tokens in vocabulary.
    In enterprise production, you use BM25 or SPLADE neural sparse models.
    """
    tokens = text.lower().replace(":", " ").replace(".", " ").split()
    counts = {}
    for t in tokens:
        if t in vocab:
            counts[vocab[t]] = counts.get(vocab[t],0) + 1.0

    # Return in Pinecone's required format: {'indices': [...], 'values': [...]}
    return {
        "indices": list(counts.keys()),
        "values": list(counts.values())
    }

def scale_hybrid_query(dense_vec: list, sparse_vec: dict, alpha: float):
    """
    Applies the Convex Combination Formula:
    Score = alpha * Dense + (1 - alpha) * Sparse
    """
    scaled_dense = [v * alpha for v in dense_vec]
    scaled_sparse = {
        "indices": sparse_vec["indices"],
        "values": [v * (1.0 - alpha) for v in sparse_vec["values"]]
    }
    return scaled_dense, scaled_sparse

def main():
    load_dotenv()
    api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from .env!")
    
    pc = Pinecone(api_key=api_key)

    index_name = "agentic-hybrid-demo"
    d = 64

    # 1. Hybrid search requires metric="dotproduct"
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Serverless Hybrid Index '{index_name}' with metric='dotproduct'...")
        pc.create_index(
            name=index_name,
            dimension=d,
            metric="dotproduct",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("Hybrid Index is READY!")
    else:
        print(f"Connecting to existing '{index_name}'...")
    index = pc.Index(index_name)

    # 2. Define a toy vocabulary for sparse token IDs
    vocab = {
        "error": 101,
        "timeout": 102,
        "err_504": 103,
        "gateway": 104,
        "database": 105,
        "network": 106,
        "latency": 107
    }

    # 3. Documents designed to show Dense vs Sparse trade-offs
    # Doc A: Highly relevant semantically to network slowness, but DOES NOT have the exact error code
    # Doc B: Has the EXACT error code 'ERR_504', but short and generic
    docs = [
        {
            "id": "doc_semantic",
            "text": "The distributed database experienced network latency and high gateway timeout issues.",
            "code": "generic_network_issue"
        },
        {
            "id": "doc_keyword_exact",
            "text": "Critical system failure: ERR_504 gateway timeout recorded in service logs.",
            "code": "ERR_504"
        }
    ]

    np.random.seed(42)
    # Generate unit-normalized dense vectors (so dot product equals cosine similarity)
    v_semantic = np.random.random(d).astype(np.float32)
    v_semantic /= np.linalg.norm(v_semantic)

    v_keyword = np.random.random(d).astype(np.float32)
    v_keyword /= np.linalg.norm(v_keyword)

    # Create Pinecone hybrid vector payloads
    vectors = [
        {
            "id": "doc_semantic",
            "values": v_semantic.tolist(),
            "sparse_values": build_sparse_vector(docs[0]["text"], vocab),
            "metadata": {"text": docs[0]["text"]}
        },
        {
            "id": "doc_keyword_exact",
            "values": v_keyword.tolist(),
            "sparse_values": build_sparse_vector(docs[1]["text"], vocab),
            "metadata": {"text": docs[1]["text"]}
        }
    ]

    print("\nUpserting hybrid vectors (dense + sparse)...")
    index.upsert(vectors=vectors, namespace="prod-hybrid")
    time.sleep(2)    # Buffer flush

    # 4. RUNNING THE HYBRID ALPHA EXPERIMENT
    print("=" * 70)
    print("BENCHMARKING ALPHA (CONVEX COMBINATION)")
    print("=" * 70)

    # Query: User asks about the exact code 'ERR_504', but semantically aligns closer to v_semantic
    query_text = "How to resolve ERR_504 timeout"
    query_dense = v_semantic.tolist()  # Artificially close to doc_semantic in dense space
    query_sparse = build_sparse_vector(query_text, vocab)

    # Test three alpha regimes:
    # 1.0 = Pure Dense
    # 0.0 = Pure Sparse
    # 0.5 = Hybrid Fusion
    alpha_settings = [1.0, 0.0, 0.5]

    for alpha in alpha_settings:
        scaled_dense, scaled_sparse = scale_hybrid_query(query_dense, query_sparse, alpha)
        res = index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=2,
            include_metadata=True,
            namespace="prod-hybrid",
        )

        mode_name = "PURE DENSE (alpha=1.0)" if alpha==1.0 else ("PURE SPARSE (alpha=0.0)" if alpha==0.0 else "HYBRID (alpha=0.5)")
        print(f"\n--- Mode: {mode_name} ---")
        for rank, match in enumerate(res.matches):
            print(f"  Rank {rank+1}: ID={match.id:<18} | Score={match.score:.4f}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY:")
    print("At alpha=1.0, semantic meaning wins. At alpha=0.0, exact keyword 'ERR_504' wins.")
    print("At alpha=0.5, hybrid fusion blends both signals for robust production RAG.")
    print("=" * 70)

if __name__ == "__main__":
    main()
