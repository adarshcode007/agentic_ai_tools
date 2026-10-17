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

### Q5: Understanding "Cosine Distance" in ChromaDB vs. Cosine Similarity in FAISS

* **Context:**
  In FAISS `IndexFlatIP`, normalized vector search returns **Cosine Similarity** (larger is closer, range: -1.0 to 1.0, where 1.0 is identical).
  In ChromaDB with `metadata={"hnsw:space": "cosine"}`, search returns **Distances** like `0.6908` and `1.0173`.

* **Questions:**
  1. What is the mathematical relationship between Chroma's Cosine Distance and Cosine Similarity?
  2. What does a distance greater than 1.0 (e.g. `1.0173`) mean geometrically in vector space?

* **Senior-Level & Architectural Answer:**
  1. **Mathematical Relationship:**
     ChromaDB uses the standard metric definition:
     `Cosine Distance = 1 - Cosine Similarity`
     * Identical vectors (angle = 0 deg): Cosine Sim = 1.0 -> Distance = 1 - 1 = 0.0 (Closest match).
     * Perpendicular / Orthogonal vectors (angle = 90 deg): Cosine Sim = 0.0 -> Distance = 1 - 0 = 1.0.
     * Opposite vectors (angle = 180 deg): Cosine Sim = -1.0 -> Distance = 1 - (-1) = 2.0 (Farthest match).
     * Rule: In Chroma, **SMALLER distance is closer** (range: 0.0 to 2.0).
  2. **Why Distance > 1.0 Occurs:**
     If Distance = 1.0173:
     `Cosine Similarity = 1 - 1.0173 = -0.0173`
     A negative similarity means the angle between the query vector and document vector is greater than 90 degrees (pointing slightly in opposite directions in semantic space).

---

### Q6: The Embedding Pipeline & Dimensionality Mismatch Trap in ChromaDB

* **Context:**
  An ingestion service ingests raw text without specifying an embedder:
  ```python
  collection.add(documents=my_docs, ids=my_ids)
  ```
  A search service later queries the collection using raw embeddings from OpenAI:
  ```python
  openai_vec = openai_client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
  results = collection.query(query_embeddings=[openai_vec], n_results=5)
  ```

* **Question:**
  What will happen at runtime when this query executes?

* **Senior-Level & Architectural Answer:**
  * **Runtime Crash (Dimensionality Mismatch):**
    The query will immediately fail with a runtime exception (e.g., `ValueError: Dimensionality of query vector (1536) does not match index dimensionality (384)`).
  * **Root Cause:**
    * When `collection.add(documents=...)` is called without a custom embedding function, Chroma initializes an underlying `hnswlib` index using its default ONNX model: `all-MiniLM-L6-v2`, which has **d = 384 dimensions**.
    * OpenAI's `text-embedding-3-small` generates vectors with **d = 1536 dimensions**.
    * An HNSW graph index is fixed to a single dimensional space at creation; computing Euclidean or Cosine distance between vectors of unequal lengths (384 vs 1536) is mathematically impossible.
  * **Production Fix:**
    Always explicitly attach the embedding function directly to the collection during initialization so both ingestion and querying stay synchronized:
    ```python
    from chromadb.utils import embedding_functions
    ef = embedding_functions.OpenAIEmbeddingFunction(api_key="...", model_name="text-embedding-3-small")
    collection = client.get_or_create_collection(name="docs", embedding_function=ef)
    ```

---

### Q7: ChromaDB Mutation Invariants (ID Collisions, `upsert`) & Flat Metadata Constraints

* **Context:**
  A production pipeline manages document lifecycle and metadata indexing in ChromaDB:
  ```python
  # Initial document
  collection.add(ids=["policy_001"], documents=["Remote work 2 days/week."], metadatas=[{"dept": "HR"}])
  ```

* **Questions:**
  1. If an engineer later runs `collection.add(ids=["policy_001"], documents=["Remote work 3 days/week."])`, what happens? How should idempotent updates be handled?
  2. If an engineer attempts to store nested structures in metadata (e.g. `metadatas=[{"approvers": ["Alice", "Bob"], "location": {"city": "NYC"}}]`), what happens?

* **Senior-Level & Architectural Answer:**
  1. **ID Collision & Idempotent Upserts:**
     * `collection.add()` strictly enforces uniqueness. If an ID already exists, Chroma raises an exception (`ValueError: DuplicateIDError / IDs already exist in collection`). It will **not** silently overwrite or duplicate.
     * **Production Fix:** Use `collection.upsert(...)` (or `collection.update(...)` if the ID is guaranteed to exist). `collection.upsert()` is idempotent: it inserts new documents if the ID is novel, or replaces the vector, text, and metadata in place if the ID already exists.
  2. **Flat Primitive Metadata Constraints:**
     * ChromaDB **rejects** nested dictionaries and lists, raising `ValueError: Expected metadata value to be a str, int, float or bool`.
     * **Why this constraint exists:** Chroma persists metadata into structured SQLite relational tables to support fast SQL `WHERE` clauses (`WHERE key = 'dept' AND string_value = 'HR'`). SQLite relational indexing is optimized for scalar primitive types, not arbitrary schemaless JSON trees.
     * **Workaround for Complex Metadata:** Serialize nested dictionaries or lists into JSON strings (e.g. `json.dumps(["Alice", "Bob"])`) or flatten the keys (e.g. `location_city="NYC"`).

---
