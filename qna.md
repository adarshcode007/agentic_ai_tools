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

