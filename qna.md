# Senior AI Engineering Q&A Knowledge Base (`qna.md`)

This living document logs architectural dilemmas, failure modes, mathematical trade-offs, and production gotchas encountered while mastering AI frameworks.

---

### Q1: The Unnormalized Vectors Trap in `IndexFlatIP` (Silent Failure Mode)

* **Context:**
  FAISS has no native `IndexFlatCosine` class. Exact Cosine Similarity is achieved by combining L2-normalized vectors ($\|v\|_2 = 1.0$) with `IndexFlatIP` (Inner Product / Dot Product):
  $$\text{Cosine}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} \implies \text{when } \|u\| = \|v\| = 1, \quad \text{Cosine}(u, v) = u \cdot v$$

* **The Dilemma:**
  An engineer indexes 50,000 document vectors `xb` into `IndexFlatIP` but **forgets** to call `faiss.normalize_L2(xb)`. At query time, incoming query vectors `xq` are properly normalized.

* **Questions:**
  1. Does FAISS throw an error or crash?
  2. How does this affect ranking? (e.g., Doc A with norm $4.0$, $\cos=0.50$ vs. Doc B with norm $1.0$, $\cos=0.99$).

* **Production & Senior-Level Answer:**
  1. **Silent Failure:** FAISS will **NOT** throw any error, warning, or exception. It executes pure, unconstrained C++ BLAS matrix multiplication. In production, this causes zero error alarms in APM/observability tools (Datadog, Langfuse) while completely corrupting retrieval quality.
  2. **Ranking Inversion:** Inner product scales with both angle and magnitude ($q \cdot d = \|q\| \|d\| \cos\theta$).
     $$\text{Score}(A) = 1.0 \times 4.0 \times 0.50 = \mathbf{2.00}$$
     $$\text{Score}(B) = 1.0 \times 1.0 \times 0.99 = \mathbf{0.99}$$
     Because $2.00 > 0.99$, Document A is ranked #1 over Document B, even though Document B is semantically identical ($8^\circ$ vs $60^\circ$ angle). Long, high-norm documents will systematically drown out highly relevant, compact documents.

* **Prevention in Production:**
  * Build an assertion/validation layer at the ingestion boundary:
    ```python
    norms = np.linalg.norm(xb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Vectors must be L2-normalized before insertion into IndexFlatIP"
    ```

---

### Q2: What Happens if Documents are Normalized, but the Query Vector is Not?

* **Context:**
  All database documents in `IndexFlatIP` are unit normalized ($\|d_i\|_2 = 1.0$). The query vector $q$ has an arbitrary norm $\|q\|_2 = c$ (e.g., $c = 3.5$).

* **The Core Phenomenon:**
  * **Relative Ranking is Unaffected:** 
    $$\text{Score}(d_i) = q \cdot d_i = \|q\|_2 \|d_i\|_2 \cos(\theta_i) = c \cdot 1.0 \cdot \cos(\theta_i) = c \cdot \cos(\theta_i)$$
    Since $c > 0$ is a constant multiplier across all documents for that query, the operation is strictly monotonic ($A > B \iff cA > cB$). Top-k ordering does not change.
  * **Score Thresholding and Filtering Completely Break:**
    Any downstream logic relying on absolute cosine thresholds (e.g., `similarity_score >= 0.75`) fails:
    - If $c > 1.0$, poor matches exceed the threshold and pollute the LLM prompt.
    - If $c < 1.0$, high-confidence matches fall below the threshold and are discarded.
  * **Cross-Query Score Incomparability:** Scores cannot be compared across different queries in a batch or cached reliably.

---

### Q3: `IndexIVFFlat` Lifecycle & Degenerative Performance (`train` vs `add` and `nprobe = nlist`)

* **Context:**
  An engineer sets up an inverted file index with $N = 100,000$ vectors partitioned across `nlist = 100` Voronoi cells:
  ```python
  quantizer = faiss.IndexFlatL2(d)
  index_ivf = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)
  ```

* **Questions:**
  1. What happens if the engineer calls `index_ivf.add(xb)` **before** `index_ivf.train(xb)`? Why does `IndexFlatL2` not require training, but `IndexIVFFlat` does?
  2. If the engineer sets `nprobe = 100` (equal to `nlist = 100`), is `index_ivf` faster or slower than `IndexFlatL2`? What does `nprobe` actually count?

* **Senior-Level & Architectural Answer:**
  1. **Lifecycle & Crash:**
     * Calling `.add()` before `.train()` raises `RuntimeError: Error in virtual void faiss::IndexIVF::add... index is not trained`.
     * **First-Principles Reason:** `IndexFlatL2` is an unpartitioned contiguous buffer; appending a vector requires zero routing logic. Conversely, `IndexIVFFlat` partitions space via K-Means centroids. If `.train()` has not run, centroids do not exist, so the quantizer cannot determine which inverted list (bucket) the incoming vector belongs to.
  2. **`nprobe` Mechanics & Degenerative Latency:**
     * **`nprobe` counts BUCKETS (Voronoi cells), NOT individual vectors.**
     * If $N = 100,000$ and `nlist = 100`, each bucket holds $\approx 1,000$ vectors.
       * `nprobe = 1` searches 1 bucket $\approx 1,000$ vectors (fastest, but boundary misses).
       * `nprobe = 5` searches 5 buckets $\approx 5,000$ vectors.
       * `nprobe = 100` opens **all 100 buckets** and scans **all 100,000 vectors**.
     * **Why `nprobe = nlist` is SLOWER than `IndexFlatL2`:**
       * `IndexFlatL2` scans 100,000 vectors stored in a single contiguous memory block using highly optimized, cache-friendly SIMD / AVX-512 BLAS matrix multiplication.
       * `IndexIVFFlat` with `nprobe = nlist` must:
         1. Compute distance to all centroids (extra compute).
         2. Traverse 100 separate inverted lists in memory (pointer chasing / memory fragmentation).
         3. Still compute distances to all 100,000 vectors anyway.
       * It does **more work with worse memory locality**, defeating the entire purpose of the index.

---

### Q4: Hardware Sizing for HNSW & Domain Drift in Product Quantization

* **Context:**
  An engineering team must index N = 10,000,000 vectors with dimension d = 1536 (OpenAI embeddings, float32) on a single cloud server with 32 GB of RAM.

* **Questions:**
  1. What is the RAM requirement for raw vectors vs. IndexHNSWFlat(d=1536, M=32)? Will it fit in 32 GB?
  2. If the team switches to IndexIVFPQ but trains the codebook on legal contracts while 99% of production data is Python code and medical papers, what happens to retrieval?

* **Senior-Level & Architectural Answer:**
  1. **Vector Memory Calculation & OOM Crash:**
     * Formula:
       Bytes per vector = d * 4 bytes (since float32 = 4 bytes)
       Bytes per vector = 1536 * 4 = 6,144 bytes (~6 KB)
       Total raw vector RAM = 10,000,000 * 6,144 bytes = 61.44 GB
     * HNSW Graph Overhead:
       HNSW stores M=32 bi-directional connection pointers per vector across multiple layers.
       This adds ~50 to 60 GB of graph pointer structures.
       Total HNSW RAM needed = ~110 GB to 120 GB.
     * Runtime Consequence on 32 GB RAM Server:
       The index cannot build. The Linux kernel OOM (Out Of Memory) Killer will terminate the Python process with exit code 137 (SIGKILL).
  2. **Codebook Domain Drift in PQ:**
     * Product Quantization is a dictionary-based lossy compressor. It learns 256 sub-centroids during training that model the geometric distribution of the training data.
     * If trained on legal contracts, the codebook sub-centroids only cluster legal terminology.
     * When Python code or medical vectors arrive, their embeddings lie in completely different regions of the 1536-dimensional space. FAISS is forced to map medical terms to the nearest legal centroid.
     * Consequence: Catastrophic quantization distortion. Cosine similarity calculations between queries and compressed vectors become meaningless, and recall plummets to near zero.
     * Production Rule: PQ training sets must be a statistically representative, diverse sample of production traffic. If data domains shift significantly, the PQ index must be retrained.

---
