# FAISS Vector Search Architectures: Executive Summary

This document synthesizes the core indexing algorithms, mathematical mechanics, benchmark trade-offs, and production selection heuristics mastered in FAISS.

---

## 1. Executive Comparison Matrix

| Index Type | Search Paradigm | Training Required? | RAM Footprint | Search Latency | Recall@k | Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`IndexFlatL2` / `IP`** | Exact Brute-Force Scan | ❌ No | Baseline (`N * d * 4` bytes) | Slowest (`O(N * d)`) | **100% (Ground Truth)** | Query Latency at scale |
| **`IndexIVFFlat`** | Voronoi Centroid Partitioning | ✅ Yes (K-Means Centroids) | Baseline (`~Flat` + small list overhead) | Fast (tuned by `nprobe`) | Good (70%–95%) | Boundary misses |
| **`IndexHNSWFlat`** | Multi-Layer Proximity Graph | ❌ No | **Highest (2.1x Baseline)** | **Blazing (< 1.5 ms)** | **Very High (95%–99%)** | DRAM / Memory cost |
| **`IndexIVFPQ`** | Sub-Space Vector Compression | ✅ Yes (Centroids + Codebook) | **Lowest (14x–32x smaller)** | Fast (table lookups) | Moderate (Lossy) | Quantization error |

---

## 2. In-Depth Architecture Breakdown

### A. `IndexFlatL2` / `IndexFlatIP` (Exact Brute-Force)

* **Mechanism:**
  * Stores uncompressed raw `float32` numbers in a contiguous C++ memory buffer.
  * At query time, computes distance against **100% of all database vectors** via hardware-accelerated SIMD / AVX-512 matrix multiplication.
* **The Cosine Invariant:**
  * FAISS has no native `IndexFlatCosine`. To compute exact Cosine Similarity, vectors must be L2-normalized in-place beforehand via `faiss.normalize_L2(xb)` and searched using `IndexFlatIP` (Inner Product).
* **Pros:**
  * **100% Recall:** Guarantees zero false negatives.
  * **Zero Training Time:** `is_trained` is `True` immediately; zero indexing latency.
* **Cons:**
  * **Linear Latency Growth:** Query complexity is `O(N * d)`. At 1M+ vectors, latency becomes unacceptable for interactive applications.
* **When to Use:**
  * Datasets under 50,000 vectors.
  * Offline batch matching jobs where accuracy is non-negotiable.
  * As the **gold-standard evaluation baseline** to benchmark recall for approximate indices.

---

### B. `IndexIVFFlat` (Inverted File / Voronoi Clustering)

* **Mechanism:**
  * Uses K-Means during `.train(xb)` to divide the vector space into `nlist` Voronoi cells, each governed by a centroid.
  * Vectors are distributed into separate inverted lists (buckets) assigned to their nearest centroid.
  * At query time, FAISS uses a coarse quantizer (`IndexFlatL2`) to find the nearest centroids to query `q`, then scans **only vectors inside the `nprobe` closest buckets**.
* **Key Invariant (`nprobe` vs. Vectors):**
  * `nprobe` specifies the number of **buckets** to open, NOT the number of vectors.
  * If `nprobe = 1`, it opens 1 bucket (fastest, but misses neighbors across cell borders).
  * If `nprobe = nlist`, it scans all buckets and performs *worse* than `IndexFlatL2` due to fragmented memory traversals.
* **Pros:**
  * 10x–30x search speedup over brute-force scan.
  * Runtime-tunable trade-off: `nprobe` can be adjusted per query without rebuilding the index.
* **Cons:**
  * Boundary misses near Voronoi borders reduce recall if `nprobe` is set too low.
  * Requires representative training data upfront (`.train()` before `.add()`).
* **When to Use:**
  * Medium-scale datasets (100k to 5M vectors) where server RAM is sufficient for raw vectors, but query latency needs a 10x–20x boost.

---

### C. `IndexHNSWFlat` (Hierarchical Navigable Small World Graph)

* **Mechanism:**
  * Constructs a multi-layer geometric proximity graph, acting as a high-dimensional skip-list.
  * Top layers contain sparse nodes with long-range "expressway" links. Layer 0 contains all `N` nodes with short-range local links.
  * Search starts at the top layer with large spatial hops, dropping down layer-by-layer until it reaches the query neighborhood on Layer 0.
* **Key Parameters:**
  * `M`: Number of bi-directional connection links per node (typically 16 to 64).
  * `efSearch`: Size of the dynamic exploration list during search (tunable at query time).
* **Pros:**
  * **The Industry Favorite:** Core algorithm powering Pinecone, Qdrant, Weaviate, and Milvus.
  * **Blazing Query Speed:** Sub-millisecond search latencies (e.g. 1.33 ms in our benchmark).
  * **Near-Perfect Recall:** Easily achieves 95%–99% recall by tuning `efSearch`.
  * **Zero Training Step:** `is_trained` is `True` immediately; supports incremental real-time additions.
* **Cons:**
  * **Extreme RAM Consumption:** Storing `M` edge pointers per node roughly doubles memory usage (`12.21 MB -> 25.19 MB` in our benchmark).
  * High build time due to graph edge calculations.
* **When to Use:**
  * Production, user-facing RAG systems where sub-second latency and high recall are mission-critical, and cloud RAM budget is available.

---

### D. `IndexIVFPQ` (Inverted File + Product Quantization)

* **Mechanism:**
  * Combines Voronoi clustering (`nlist`) with **Product Quantization (PQ)** vector compression.
  * Slices each `d`-dimensional vector into `m` smaller sub-vectors (e.g. 64 dims / 8 = 8-dim slices).
  * Runs K-Means on each sub-space to find 256 sub-centroids (`nbits = 8`, since `2^8 = 256`).
  * Replaces each sub-vector with the **1-byte integer ID** of its nearest sub-centroid, compressing 64 floats (256 bytes) down to **8 bytes**.
  * Query distances are calculated via Asymmetric Distance Computation (ADC) using precomputed lookup tables.
* **The Codebook Drift Failure Mode:**
  * The codebook centroids reflect the statistical distribution of the training data. If trained on legal contracts, the codebook cannot accurately represent code or medical embeddings, causing catastrophic quantization error and near-zero recall.
* **Pros:**
  * **Massive RAM Compression:** Compresses memory footprint by 14x–32x (`12.21 MB -> 0.85 MB` in our benchmark).
  * Fast query execution via lookup tables.
* **Cons:**
  * **Lossy Compression:** Quantization introduces distance approximation errors, lowering precision.
  * Heavy build/train time across sub-space K-Means.
* **When to Use:**
  * Massive datasets (10M to 1B+ vectors) where RAM hardware costs are the primary bottleneck.
  * Ideal as a **Stage-1 candidate generator** (over-fetching candidates), paired with a local Cross-Encoder re-ranker in Stage 2.

---

## 3. Production Decision Tree

```
How many vectors do you need to search?
│
├── < 50,000 vectors
│     └── Use IndexFlatIP (with normalized vectors) -> Simple, exact, zero overhead.
│
├── 50,000 to 5,000,000 vectors
│     │
│     ├── Do you have sufficient RAM budget?
│     │     ├── Yes -> Use IndexHNSWFlat -> Sub-millisecond latency, >95% recall.
│     │     └── No  -> Use IndexIVFFlat  -> Good speedup, standard RAM, tune via nprobe.
│
└── > 5,000,000 vectors
      │
      ├── Memory-constrained environment?
      │     └── Use IndexIVFPQ -> 14x-32x memory compression + Cross-Encoder re-ranking.
      │
      └── Enterprise scale with dedicated cluster infrastructure?
            └── Use Distributed HNSW (e.g. Qdrant / Pinecone) or HNSW on SSD/DiskANN.
```
