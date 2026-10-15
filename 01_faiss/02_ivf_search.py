import time
import numpy as np 
import faiss 

def calculate_recall(ground_truth_ids: np.ndarray, approx_ids: np.ndarray) -> float:
    """
    Computes average Recall@k:
    Fraction of true top-k nearest neighbors retrieved by the approximate index.
    """
    nq, k = ground_truth_ids.shape
    recalls = []
    for i in range(nq):
        # Count intersection between exact top-k and approx top-k
        true_set = set(ground_truth_ids[i])
        approx_set = set(approx_ids[i])
        intersection = true_set.intersection(approx_set)
        recalls.append(len(intersection) / k)
    return float(np.mean(recalls))

def main():
    np.random.seed(42)

    # 1. Dataset sizing: 100,000 vectors of dimension 64
    d = 64
    N = 100000
    nq = 100    # 100 query vectors to get statistically meaningful latency/recall
    k = 10      # Retrieve top-10 neighbors

    print(f"Generating dataset: N={N:,} vectors, d={d}, nq={nq} queries...")
    xb = np.random.random((N, d)).astype(np.float32)
    xq = np.random.random((nq, d)).astype(np.float32)


    # PART 1: GROUND TRUTH BENCHMARK (Exact IndexFlatL2)
    print("\n" + "=" * 65)
    print("PART 1: EXACT BRUTE-FORCE BASELINE (IndexFlatL2)")
    print("=" * 65)

    index_flat = faiss.IndexFlatL2(d)
    index_flat.add(xb)

    t0 = time.perf_counter()
    D_exact, I_exact = index_flat.search(xq, k)
    exact_time_ms = (time.perf_counter() - t0) * 1000

    print(f"IndexFlatL2 Search Latency for {nq} queries: {exact_time_ms:.2f} ms")
    print(f"Ground Truth Recall: 100.00% (by definition)")

    # PART 2: APPROXIMATE SEARCH (IndexIVFFlat)
    print("\n" + "=" * 65)
    print("PART 2: APPROXIMATE INVERTED FILE SEARCH (IndexIVFFlat)")
    print("=" * 65)

    # nlist: Number of Voronoi cells (clusters).
    # Rule of thumb: 4 * sqrt(N) to 16 * sqrt(N). For 100k, sqrt(100k) ~ 316.
    nlist = 100

    # The 'quantizer' is the index used to assign vectors to Voronoi centroids.
    # We use a simple FlatL2 index to find which centroid is closest.
    quantizer = faiss.IndexFlatL2(d)

    # Instantiate IndexIVFFlat
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    # Notice: is_trained is False before training!
    print(f"Before training - is_trained: {index_ivf.is_trained}")

    # Training step: Runs K-means to find the 100 Voronoi centroids
    t_train = time.perf_counter()
    index_ivf.train(xb)
    train_time_ms = (time.perf_counter() - t_train) * 1000
    print(f"After training  - is_trained: {index_ivf.is_trained} (trained in {train_time_ms:.2f} ms)")
    
    # Adding vectors: assigns each vector into one of the 100 inverted lists
    index_ivf.add(xb)
    print(f"Total vectors added to IVF index: {index_ivf.ntotal}")


    # PART 3: THE LATENCY VS RECALL TRADE-OFF (TUNING nprobe)
    print("\n" + "=" * 65)
    print("PART 3: BENCHMARKING nprobe (Search Depth vs. Recall vs. Latency)")
    print("=" * 65)
    print(f"{'nprobe':<10} | {'Latency (ms)':<15} | {'Speedup':<12} | {'Recall@10':<12}")
    print("-" * 65)

    # Test different nprobe values (cells to search)
    nprobe_values = [1, 2, 5, 10, 20, 50, 100]

    for nprobe in nprobe_values:
        # Set how many Voronoi cells to visit during search
        index_ivf.nprobe = nprobe

        t_search = time.perf_counter()
        D_ivf, I_ivf = index_ivf.search(xq, k)
        ivf_time_ms = (time.perf_counter() - t_search) * 1000

        # Calculate accuracy compared to exact ground truth
        recall = calculate_recall(I_exact, I_ivf)
        speedup = exact_time_ms / ivf_time_ms

        print(f"{nprobe:<10} | {ivf_time_ms:<15.2f} | {speedup:<12.2f}x | {recall * 100:<11.2f}%")

    print("=" * 65)
    print("KEY TAKEAWAYS:")
    print("1. nprobe=1 is the fastest, but misses neighbors sitting across cell boundaries.")
    print("2. As nprobe approaches nlist (100), recall hits ~100%, but speed advantage disappears.")
    print("3. In production, choose the lowest nprobe that satisfies your SLA (e.g. >=95% recall).")

if __name__ == "__main__":
    main()  

