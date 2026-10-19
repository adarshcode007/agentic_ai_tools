"""
01_embeddings_and_reranking.py
Phase 1: Hugginf Face Ecosystem & Local Model Deployment
Subtopic: Open-Source Embeddings (Bi-Encoder) vs Cross-Encoder Re-Ranking
"""

import time
import numpy as np 
from sentence_transformers import SentenceTransformer, CrossEncoder

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    # PART 1: LOAD MODELS
    print_banner("PART 1: LOADING LOCAL MODELS FROM HUGGING FACE")
    print("Loading Bi-Encoder: 'all-MiniLM-L6-v2' (Dimension: 384)...")
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Note: Model name is 'ms-marco' (Microsoft MARCO passage dataset)
    print("Loading Cross-Encoder: 'cross-encoder/ms-marco-MiniLM-L-6-v2'...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Models loaded successfully into local RAM/VRAM!")


    # PART 2: CRAFT THE CORPUS (Testing Semantic vs. Keyword Blind Spots)
    print_banner("PART 2: CORPUS & QUERY DEFINITION")

    query = "How to safely recover a lost root password on Linux"
    print(f"Target Query: '{query}'\n")

    corpus = [
        {
            "id": "doc_keyword_distractor",
            "type": "Keyword Distractor (High word overlap, wrong action)",
            "text": (
                "To change your regular user password on Linux, open a terminal "
                "and run the standard passwd command without root privileges."
            ),
        },
        {
            "id": "doc_semantic_distractor",
            "type": "Semantic Distractor (Relates to root, but no recovery steps)",
            "text": (
                "The Linux root account is the master administrative superuser who "
                "possesses unrestricted read and write permissions to all system files."
            ),
        },
        {
            "id": "doc_golden_answer",
            "type": "Golden Answer (Low word overlap, true technical solution)",
            "text": (
                "To regain administrative access on Ubuntu, reboot into the GRUB "
                "bootloader, enter single-user recovery mode, and mount the root "
                "filesystem as read-write to reset the superuser credentials."
            ),
        },
        {
            "id": "doc_irrelevant",
            "type": "Irrelevant Baseline",
            "text": (
                "The authentic Neapolitan pizza recipe requires high-gluten flour, "
                "San Marzano tomatoes, and extra-virgin olive oil fermented for 24 hours."
            ),
        },
    ]

    for doc in corpus:
        print(f"[{doc['id']}] ({doc['type']}):\n -> \" {doc['text']}\"\n")

    
    # PART 3: STAGE 1- BI-ENCODER RETRIEVAL (Dense Cosine)
    print_banner("PART 3: STAGE 1 — BI-ENCODER RETRIEVAL (Dense Cosine)")
    doc_texts = [d["text"] for d in corpus]

    # Normalize vectors so dot product directly equals Cosine Similarity
    doc_embeddings = bi_encoder.encode(doc_texts,normalize_embeddings=True)
    query_embedding = bi_encoder.encode(query, normalize_embeddings=True)

    # Compute dot products: (N, d) . (d,) -> (N,)
    bi_scores = np.dot(doc_embeddings, query_embedding)

    # Rank documents by Bi-Encoder score descending
    stage1_ranked_indices = np.argsort(bi_scores)[::-1]

    print(f"{'Rank':<6} | {'Doc ID':<25} | {'Bi-Score':<10} | {'Document Excerpt'}")
    print("-" * 75)

    for rank, idx in enumerate(stage1_ranked_indices, start=1):
        doc = corpus[idx]
        print(f"#{rank:<5} | {doc['id']:<25} | {bi_scores[idx]:.4f} | {doc['text'][:45]}...")


    # PART 4: STAGE 2- CROSS-ENCODER RE-RANKING (Joint Cross-Attention)
    print_banner("PART 4: STAGE 2 - CROSS-ENCODER RE-RANKING (Joint Attention)")

    # Prepare pairs: [CLS] query [SEP] document [SEP]
    pairs = [[query, doc["text"]] for doc in corpus]
    cross_scores = cross_encoder.predict(pairs)

    # Rank documents by Cross-Encoder score descending
    stage2_ranked_indices = np.argsort(cross_scores)[::-1]

    print(f"{'Rank':<6} | {'Doc ID':<25} | {'CE-Score':<10} | {'Document Excerpt'}")
    print("-" * 75)
    for rank, idx in enumerate(stage2_ranked_indices, start=1):
        doc = corpus[idx]
        print(f"#{rank:<5} | {doc['id']:<25} | {cross_scores[idx]:.4f}     | {doc['text'][:45]}...")

    
    # PART 5: COMPARISON & THE RANKING INVERSION
    print_banner("PART 5: RETRIEVAL COMPARISON & RANKING INVERSION")
    golden_idx = [i for i, d in enumerate(corpus) if d["id"] == "doc_golden_answer"][0]
    stage1_golden_rank = list(stage1_ranked_indices).index(golden_idx) + 1
    stage2_golden_rank = list(stage2_ranked_indices).index(golden_idx) + 1
    print(f"Target Document: '{corpus[golden_idx]['id']}'")
    print(f"  * Stage 1 (Bi-Encoder) Rank   : #{stage1_golden_rank} (Score: {bi_scores[golden_idx]:.4f})")
    print(f"  * Stage 2 (Cross-Encoder) Rank: #{stage2_golden_rank} (Score: {cross_scores[golden_idx]:.4f})")
    if stage2_golden_rank < stage1_golden_rank:
        print("\n[SUCCESS] RANKING INVERSION DETECTED!")
        print(f"  The Cross-Encoder elevated the Golden Answer from #{stage1_golden_rank} -> #{stage2_golden_rank}.")
        print("  Why? The Bi-Encoder was deceived by superficial keyword overlaps in distractor docs.")
        print("  The Cross-Encoder's multi-head cross-attention evaluated the actual semantic relationship.")

    # PART 6: LATENCY & COMPUTATIONAL COMPLEXITY BENCHMARK
    print_banner("PART 6: COMPUTATIONAL COST & LATENCY BENCHMARK")
    iterations = 50

    # 1. Bi-Encoder Dot Product Latency (Stage 1 search)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = np.dot(doc_embeddings, query_embedding)
    bi_dot_time = (time.perf_counter() - t0) / iterations * 1000

    # 2. Cross-Encoder Forward Pass Latency (Stage 2 re-ranking)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = cross_encoder.predict(pairs)
    ce_predict_time = (time.perf_counter() - t0) / iterations * 1000

    print(f"Benchmark over {iterations} iterations:")
    print(f"  * Bi-Encoder Dot Product Time: {bi_dot_time:.5f} ms per query")
    print(f"  * Cross-Encoder Predict Time : {ce_predict_time:.2f} ms per query ({len(corpus)} docs)")
    print(f"  * Latency Ratio             : Cross-Encoder is ~{ce_predict_time / max(bi_dot_time, 1e-6):.1f}x heavier!")
    print("\nArchitectural Conclusion:")
    print("  -> Use Bi-Encoder (or Vector DB index) to prune 10,000,000 docs down to Top-25.")
    print("  -> Use Cross-Encoder only on those Top-25 candidates for precision re-ranking.")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
    