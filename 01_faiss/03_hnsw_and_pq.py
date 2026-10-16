import os
import time
import numpy as np 
import faiss

def calculate_recall(ground_truth_ids: np.ndarray, approx_ids: np.ndarray) -> float:
    nq, k = ground_truth_ids.shape
    recalls = []
    for i in range(nq):
        true_set = set(ground_truth_ids[i])
        approx_set = set(approx_ids[i])
        recalls.append(len(true_set.intersection(approx_set)) / k)
    return float(np.mean(recalls))

def get_index_size_mb(index, filename="temp_index.faiss") -> float:
    """Saves index to disk temporarily to measure exact serialized byte size."""
    faiss.write_index(index, filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    if os.path.exists(filename):
        os.remove(filename)
    return size_mb

def main():
    np.random.seed(42)
    d = 64
    N = 50000
    nq = 100
    k = 10

    print(f"Generating dataset: N={N:,} vectors, d={d}, nq={nq} queries...")
    xb = np.random.random((N,d)).astype(np.float32)
    xq = np.random.random((nq,d)).astype(np.float32)

    # 1. BASELINE: IndexFlatL2
    print("\n--- 1. IndexFlatL2 (Exact Baseline) ---")
    index_flat = faiss.IndexFlatL2(d)

    t0 = time.perf_counter()
    index_flat.add(xb)
    build_flat_ms = (time.perf_counter() - t0) * 1000

    to = time.perf_counter()
    D_exact, I_exact = index_flat.search(xq, k)
    search_flat_ms = (time.perf_counter() - t0) * 1000
    size_flat_mb = get_index_size_mb(index_flat)

    # 2. GRAPH-BASED: IndexHNSWFlat
    print("\n--- 2. IndexHNSWFlat (Hierarchical Navigable Small World) ---")
    M = 32   # Number of bi-directional connection links per node
    index_hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)

    print(f"HNSW is_trained before add: {index_hnsw.is_trained}")  # True! No clustering needed.
    
    t0 = time.perf_counter()
    index_hnsw.add(xb)  # Builds the multi-layer graph
    build_hnsw_ms = (time.perf_counter() - t0) * 1000

    # efSearch: Size of the dynamic candidate exploration list at search time
    index_hnsw.hnsw.efSearch = 64

    t0 = time.perf_counter()
    D_hnsw, I_hnsw = index_hnsw.search(xq, k)
    search_hnsw_ms = (time.perf_counter() - t0) * 1000
    size_hnsw_mb = get_index_size_mb(index_hnsw)
    recall_hnsw = calculate_recall(I_exact, I_hnsw)

    # 3. COMPRESSED: IndexIVFPQ (Inverted File + Product Quantization)
    print("\n--- 3. IndexIVFPQ (Voronoi Buckets + Byte Compression) ---")
    nlist = 100     # 100 Voronoi cells
    m = 8           # Number of sub-vector segments (64 dims / 8 = 8-dim subvectors)
    nbits = 8       # 8 bits per sub-vector code = 256 centroids per sub-space (1 byte per code)

    quantizer = faiss.IndexFlatL2(d)
    index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    print(f"IVFPQ is_trained before training: {index_ivfpq.is_trained}")  # False

    t0 = time.perf_counter()
    index_ivfpq.train(xb)   # Trains both Voronoi centroids AND PQ codebook
    train_ivfpq_ms = (time.perf_counter() - t0) * 1000

    index_ivfpq.add(xb)    # Quantizes vectors into compressed byte codes
    build_ivfpq_ms = train_ivfpq_ms + (time.perf_counter() - t0) * 1000

    index_ivfpq.nprobe = 10  # Inspect 10 cells

    t0 = time.perf_counter()
    D_ivfpq, I_ivfpq = index_ivfpq.search(xq, k)
    search_ivfpq_ms = (time.perf_counter() - t0) * 1000
    size_ivfpq_mb = get_index_size_mb(index_ivfpq)
    recall_ivfpq = calculate_recall(I_exact, I_ivfpq)


    # PRODUCTION COMPARISON TABLE
    print("\n" + "=" * 80)
    print(f"{'Index Type':<16} | {'Size (MB)':<10} | {'Build Time':<12} | {'Search Latency':<15} | {'Recall@10':<10}")
    print("=" * 80)
    print(f"{'IndexFlatL2':<16} | {size_flat_mb:<10.2f} | {build_flat_ms:<9.2f} ms | {search_flat_ms:<12.2f} ms | {'100.00%':<10}")
    print(f"{'IndexHNSWFlat':<16} | {size_hnsw_mb:<10.2f} | {build_hnsw_ms:<9.2f} ms | {search_hnsw_ms:<12.2f} ms | {recall_hnsw * 100:<9.2f}%")
    print(f"{'IndexIVFPQ':<16} | {size_ivfpq_mb:<10.2f} | {build_ivfpq_ms:<9.2f} ms | {search_ivfpq_ms:<12.2f} ms | {recall_ivfpq * 100:<9.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
