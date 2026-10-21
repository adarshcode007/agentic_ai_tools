"""
01_vector_and_summary_indices.py
======================================================================
LlamaIndex Core Index Architectures: VectorStoreIndex vs. SummaryIndex
======================================================================
Demonstrates:
  1. Decoupling Embeddings and LLMs using global Settings (local Hugging Face embeddings).
  2. Building a VectorStoreIndex and inspecting NodeWithScore retrieval results.
  3. Building a SummaryIndex (zero-embedding sequential traversal for synthesis).
  4. Persisting StorageContext to disk (docstore.json, default__vector_store.json)
     and zero-recomputation reloading.
"""

import shutil
from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import MockLLM

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    # PART 1: CONFIGURE GLOBAL SETTINGS (Local Open-Source Embeddings)
    print_banner("PART 1: CONFIGURING GLOBAL SETTINGS")
    
    print("Initializing local HuggingFaceEmbedding ('sentence-transformers/all-MiniLM-L6-v2')...")
    # Runs 100% locally on CPU/GPU with zero external API calls or fees
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use MockLLM so we can test index mechanics and retrieval without needing paid API keys
    Settings.llm = MockLLM()
    print("Settings initialized successfully.")

    # Ingest the sample documents from the previous module
    data_dir = Path(__file__).parent.parent / "09_llamaindex_ingestion" / "data"
    print(f"\nLoading documents from: {data_dir}")
    reader = SimpleDirectoryReader(input_dir=str(data_dir))
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document(s) for indexing.")

    # PART 2: VectorStoreIndex (DENSE VECTOR RETRIEVAL)
    print_banner("PART 2: VectorStoreIndex (SEMANTIC SEARCH)")

    print("Building VectorStoreIndex (calculating embeddings for all chunks)...")
    vector_index = VectorStoreIndex.from_documents(documents)
    print("VectorStoreIndex constructed.")

    # Convert the index into a pure Retriever (fetches nodes without calling LLM)
    vector_retriever = vector_index.as_retriever(similarity_top_k=2)

    query = "What is the Recovery Point Objective (RPO) and Recovery Time Objective?"
    print(f"\nTarget Query: \"{query}\"")
    print("Executing Vector Search...")

    retrieved_nodes = vector_retriever.retrieve(query)
    print(f"\nRetrieved {len(retrieved_nodes)} node(s) via Vector Similarity:")
    print("-" * 70)

    for i, scored_node in enumerate(retrieved_nodes):
        print(f"[Result {i+1}] Similarity Score: {scored_node.score:.4f}")
        print(f"Source File     : {scored_node.node.metadata.get('file_name')}")
        print(f"Text Snippet    : {scored_node.node.text[:100].replace(chr(10), ' ')}...")
        print("-" * 70)
    
    # PART 3: SummaryIndex (SEQUENTIAL ALL-NODE TRAVERSAL)
    print_banner("PART 3: SummaryIndex (WHOLE-CORPUS SYNTHESIS)")

    print("Building SummaryIndex (Notice: ZERO embeddings computed)...")
    summary_index = SummaryIndex.from_documents(documents)
    print("SummaryIndex constructed.")

    summary_retriever = summary_index.as_retriever()
    summary_nodes = summary_retriever.retrieve("Summarize all security policies")

    print(f"\nSummaryIndex Retrieved {len(summary_nodes)} node(s):")
    print(f"  * Total Nodes in Index : {len(summary_nodes)}")
    print("  * Notice: SummaryIndex retrieved EVERY node in the corpus sequentially!")
    print("  * Best for: Comprehensive summaries where vector search misses distant sections.")

    # PART 4: PERSISTENCE & ZERO-COST RELOADING
    print_banner("PART 4: STORAGE CONTEXT PERSISTENCE & RELOAD")

    storage_dir = Path(__file__).parent / "storage"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    
    print(f"Persisting VectorStoreIndex to disk at: '{storage_dir}'...")
    vector_index.storage_context.persist(persist_dir=str(storage_dir))

    print("\nFiles generated on disk:")
    for f in storage_dir.iterdir():
        print(f"  ├── {f.name:<30} ({f.stat().st_size:,} bytes)")

    print("\nDemonstrating Cold-Start Reload (Zero Re-Embedding):")
    # Reload the index directly from the persisted JSON files into VectorStoreIndex in memory
    reloaded_storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    reloaded_index = load_index_from_storage(reloaded_storage_context)
    print("Index reloaded from disk successfully!")

    reloaded_retriever = reloaded_index.as_retriever(similarity_top_k=1)
    reloaded_result = reloaded_retriever.retrieve(query)
    print(f"\nVerification Query on Reloaded Index:")
    print(f"  * Top Score : {reloaded_result[0].score:.4f}")
    print(f"  * Top Match : {reloaded_result[0].node.text[:80].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    main()


