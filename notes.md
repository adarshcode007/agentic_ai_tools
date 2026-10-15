# Vector Databases & Search Engines: Deep-Dive Notes (`notes.md`)

This document contains deep architectural explanations, internal mechanics, and mental models of vector search algorithms and data structures.

---

## 1. What is a "Quantizer" in FAISS?

### The Core Problem:
An Inverted File index (`IndexIVFFlat`) partitions a continuous vector space into geometric buckets called **Voronoi Cells**. 
Each cell has a center point called a **Centroid** ($\mu_1, \mu_2, \dots, \mu_{nlist}$).

When a vector arrives—either during `.add(xb)` to store it, or during `.search(xq)` to query it—FAISS faces a sub-problem:
> *"Out of the `nlist` centroids, which one(s) are physically closest to this vector?"*

Finding the closest centroid to a vector is, by definition, a **nearest neighbor search problem**. 

The component that performs this centroid lookup is called the **Coarse Quantizer**:
* **Why the word "Quantizer"?** In signal processing and information theory, *quantization* means mapping continuous values to a finite set of discrete symbols. Here, the quantizer maps a continuous 64-dimensional float vector to a discrete integer bucket ID (`0` to `nlist - 1`).

---

## 2. Why is the Quantizer Instantiated with `IndexFlatL2`?

In `02_ivf_search.py`, we wrote:
```python
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
```

### Why use `IndexFlatL2` for an IVF index?

1. **Centroid Count is Tiny:**
   * Your dataset has $N = 100,000$ (or millions of) vectors.
   * But `nlist` (the number of centroids) is tiny—usually between **$100$ and $4,096$**.
   * Running an exact brute-force scan (`IndexFlatL2`) over only $100$ centroid points takes fractions of a microsecond.
2. **Guaranteed Accuracy for Bucket Assignment:**
   * You want the index to accurately determine which Voronoi cell the vector belongs to without making approximation errors at the routing stage.
3. **Decoupled Architecture (FAISS Modularity):**
   * FAISS separates the **routing engine** (the quantizer) from the **storage engine** (the inverted lists).
   * For standard databases, `IndexFlatL2` is the ideal coarse quantizer.
   * *(Advanced Architecture)*: In multi-billion vector datasets where `nlist = 65,536` or higher, searching centroids with `IndexFlatL2` becomes a bottleneck. In those extreme cases, engineers swap `IndexFlatL2` with a graph index like `IndexHNSWFlat` as the quantizer!

---

## 3. Summary of Data Flow:

```
[Incoming Vector]
       │
       ▼
[Quantizer: IndexFlatL2]  ──▶ (Scans 100 Centroids with exact L2)
       │
       ▼ Returns: Closest Centroid ID = #42
[Inverted List #42]       ──▶ (Appends vector ID into Bucket 42)
```

---

## 4. What is `nprobe` in FAISS?

### Definition:
`nprobe` (number of probes) is the **search-time parameter** that specifies **how many Voronoi cells (centroid buckets) FAISS will inspect** when answering a query.

### The Boundary Problem (Why `nprobe = 1` misses data):
* Space is partitioned into Voronoi cells. When a query vector $q$ lands near the **boundary** of its cell, its true nearest neighbors might sit just millimeters across the border inside an adjacent cell.
* If `nprobe = 1`, FAISS *only* searches the single closest cell and completely ignores all neighboring cells, resulting in low recall (as seen in our benchmark: **10.8% Recall@10**).

### How FAISS Uses `nprobe`:
1. When query $q$ arrives, the coarse quantizer sorts all centroids by distance to $q$.
2. FAISS picks the **top `nprobe` closest centroids**.
3. It opens only those `nprobe` inverted lists and scans the vectors inside them.

### The Latency vs. Recall Dial:
| `nprobe` Setting | Behavior | Latency | Recall |
| :--- | :--- | :--- | :--- |
| `nprobe = 1` | Inspects only 1 cell | Blazing fast (15x–30x speedup) | Poor (frequent boundary misses) |
| `1 < nprobe < nlist` | Inspects neighborhood of closest cells | Balanced | High ($\ge 90\%-98\%$) |
| `nprobe = nlist` | Inspects all cells | Slowest (same work as `IndexFlatL2`) | 100% (degenerates into exact scan) |

> **Production Rule of Thumb:** `nprobe` is tuned at query runtime without rebuilding the index. You increase `nprobe` until your retrieval meets your accuracy SLA (e.g., 95% Recall@10).

