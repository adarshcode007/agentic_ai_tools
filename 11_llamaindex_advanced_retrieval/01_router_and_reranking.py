"""
01_router_and_reranking.py
======================================================================
Advanced Query Engines: Dynamic Routing + Cross-Encoder Re-Ranking
======================================================================
Demonstrates:
  1. Groq LLM integration ('openai/gpt-oss-20b') for blazing-fast RAG.
  2. Local dense embeddings (all-MiniLM-L6-v2) + local Cross-Encoder (ms-marco-MiniLM-L-6-v2).
  3. Two-stage retrieval: Over-fetching top-5 candidates -> Cross-encoder re-ranking top-2.
  4. Dynamic RouterQueryEngine with LLMSingleSelector to route between vector search & summary.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    Settings,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    # PART 1: CONFIGURE GROQ LLM & LOCAL HUGGING FACE MODELS
    print_banner("PART 1: INITIALIZING GROQ & EMBEDDING MODELS")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY missing from .env! Please add it before running.")
    
    target_model = "openai/gpt-oss-20b"
    print(f"Initializing Groq LLM (Model: '{target_model}')...")

    # Initialize Groq LLM
    llm = Groq(model=target_model, api_key=groq_api_key, temperature=0.1) 
    
    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Groq LLM and HuggingFaceEmbedding initialized successfully.")

    # PART 2: BUILD DUAL INDICES (VectorStoreIndex + SummaryIndex)
    print_banner("PART 2: INGESTION & DUAL INDEX CREATION")
    
    data_dir = Path(__file__).parent.parent / "09_llamaindex_ingestion" / "data"
    documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()
    print(f"Loaded {len(documents)} document(s) from {data_dir.name}.")

    print("Building VectorStoreIndex...")
    vector_index = VectorStoreIndex.from_documents(documents)

    print("Building SummaryIndex...")
    summary_index = SummaryIndex.from_documents(documents)

    # PART 3: CONFIGURE TWO-STAGE RETRIEVAL WITH RERANKING
    print_banner("PART 3: TWO-STAGE RETRIEVAL (OVER-FETCH + CROSS-ENCODER)")

    # 1. Similarity cutoff: discard completely irrelevant chunks (< 0.20 score)
    cutoff_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.20)

    # 2. Local Cross-Encoder Re-ranker: ms-marco-MiniLM-L-6-v2 evaluates (query, text) pairs
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=2 # Compress top-5 candidates down to top-2 highest cross-attention matches
    )

    # Build the Vector Query Engine with the postprocessor pipeline
    vector_query_engine = vector_index.as_query_engine(
        similarity_top_k=5, # Stage 1: Over-fetch 5 candidates
        node_postprocessors=[cutoff_postprocessor, reranker] # Stage 2: Filter and Re-rank
    )

    summary_query_engine = summary_index.as_query_engine()

    # PART 4: DYNAMIC ROUTER SETUP (RouterQueryEngine)
    print_banner("PART 4: BUILDING DYNAMIC RouterQueryEngine")

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific facts, technical parameters, "
            "RTO/RPO targets, SLA numbers, and specific architectural details."
        )
    )

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for high-level summaries, holistic overviews, and general "
            "syntheses across all policies and architecture documents."
        )
    )

    # LLMSingleSelector uses Groq to classify the user's intent into 1 tool
    router_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[vector_tool, summary_tool],
        verbose=True # Prints internal routing choices in real-time!
    )
    print("RouterQueryEngine assembled with vector_tool and summary_tool.")

    # PART 5: LIVE TEST 1 — PINPOINT FACT RETRIEVAL (Should pick Vector)
    print_banner("TEST 1: PINPOINT FACT LOOKUP")
    q1 = "What is the exact Recovery Point Objective (RPO) and Recovery Time Objective (RTO)?"
    print(f"User Query: \"{q1}\"\n")
    
    response_1 = router_engine.query(q1)
    print("\n--- Groq Final Answer ---")
    print(response_1.response)
    print("-" * 70)
    print(f"Source Nodes Used: {len(response_1.source_nodes)}")
    for node in response_1.source_nodes:
        print(f"  * Node File: {node.metadata.get('file_name')} | Score: {node.score:.4f}")

    # PART 6: LIVE TEST 2 — HIGH-LEVEL SUMMARY (Should pick Summary)
    print_banner("TEST 2: WHOLE-DOCUMENT SUMMARY")
    q2 = "Provide a high-level executive summary of our compliance rules and architecture."
    print(f"User Query: \"{q2}\"\n")
    
    response_2 = router_engine.query(q2)
    print("\n--- Groq Final Answer ---")
    print(response_2.response)
    print("-" * 70)
    print(f"Source Nodes Used: {len(response_2.source_nodes)}")
    for node in response_2.source_nodes:
        print(f"  * Node File: {node.metadata.get('file_name')}")
    
if __name__ == "__main__":
    main()
