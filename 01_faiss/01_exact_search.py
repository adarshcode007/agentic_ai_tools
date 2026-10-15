import numpy as np
import faiss

def main():
    # Set seed for reproducible synthetic data
    np.random.seed(42)

    # 1. Dimensionality and Dataset sizing
    # d: vector dimension (eg. 64 for testing, 1536 for OpenAI, 384 for miniLM)
    # N: number of database vectors
    # nq: number of incoming query vectors
    d = 64
    N = 10000
    nq = 5

    # CRITICAL FAISS INVARIANT:
    # All vectors MUST be contigous float32 numpy arrays.
    # 'xb' = database vectors, 'xq' = query vectors (FAISS naming convention)
    xb = np.random.random((N,d)).astype(np.float32)
    xq = np.random.random((nq, d)).astype(np.float32)

    # In real applications, FAISS has No concept of text or metadata
    # You maintain an external mapping of: integer ID -> document content/metdata
    doc_store = {i: f"Document_Chunk_{i}" for i in range(N)}

    print("="*60)
    print("PART 1: EXACT EUCLIDEAN (L2) SEARCH - IndexFlatL2")
    print("="*60)

    # 2. Instantiate IndexFlatL2
    index_l2 = faiss.IndexFlatL2(d)

    # Flat indices do NOT require training because they don't cluster or quantize.
    # They simply store the raw vectors in an uncompressed memory buffer.
    print(f"Is index trained? {index_l2.is_trained}")  # Always True for Flat

    # Add database vectors to the index
    index_l2.add(xb)
    print(f"Total vectors stored in index: {index_l2.ntotal}")

    # 3. Search: Retrieve top-k nearest neighbors
    k = 3
    # D: Distances (shape: [nq, k]), squared Euclidean distances ||q - v||^2
    # I: Indices/IDs (shape: [nq, k]), row index in xb (0 to N-1)
    D_l2, I_l2 = index_l2.search(xq, k) 
    

    print("\nL2 Search Results for Query 0:")
    for rank in range(k):
        neighbor_id = I_l2[0][rank]
        dist = D_l2[0][rank]
        print(f"  Rank {rank+1}: ID={neighbor_id} ({doc_store[neighbor_id]}), Distance={dist:.4f}")
    
    print("\n"+"="*60)
    print("PART 2: EXACT COSINE SIMILARITY - IndexFlatIP")
    print("="*60)

    # 4. Preparing for Cosine Similarity
    # FAISS has no "IndexFlatCosine"
    # Cosine Similarity is Inner Product ON NORMALIZED VECTORS:
    #  Cosine(u,v) = (u . v) / (||u|| * ||v||)
    #  If ||u||=1 and ||v||=1, then Cosine(u,v) = u . v
    xb_norm = xb.copy()
    xq_norm = xq.copy()

    # In-place L2 normalization: divides each vector by its Euclidean norm
    faiss.normalize_L2(xb_norm)
    faiss.normalize_L2(xq_norm)

    # Verfication: check that norm of the first vector is now 1.0
    norm_check = np.linalg.norm(xb_norm[0])
    print(f"Normalized vector L2 norm check: {norm_check:.6f}")

    # 5. Instantiate IndexFlatIP (Inner Product)
    index_ip = faiss.IndexFlatIP(d)
    index_ip.add(xb_norm)
    print(f"Total vectors in IP index: {index_ip.ntotal}")

    # 6. Search with normalized queries
    D_ip, I_ip = index_ip.search(xq_norm, k)

    print("\nCosine (IP) Search Results for Query 0:")
    for rank in range(k):
        neighbor_id = I_ip[0][rank]
        similarity = D_ip[0][rank]  # Cosine similarity in range [-1.0, 1.0]
        print(f"   Rank {rank+1}: ID={neighbor_id} ({doc_store[neighbor_id]}), CosineSim={similarity:.4f}")

    print("\n" + "=" * 60)
    print("KEY METRIC COMPARISON:")
    print("  IndexFlatL2: SMALLER is closer (0.0 = identical).")
    print("  IndexFlatIP: LARGER is closer  (1.0 = identical direction).")
    print("=" * 60)

if __name__ == "__main__":
    main()