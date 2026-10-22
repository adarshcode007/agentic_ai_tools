"""
02_subquestion_and_hyde.py
======================================================================
Query Decomposition (SubQuestionQueryEngine) & HyDE Query Transforms
======================================================================
Demonstrates:
  1. Decomposing multi-part questions across separate document tools
     using SubQuestionQueryEngine.
  2. Transforming abstract queries into rich embeddings using HyDEQueryTransform
     and TransformQueryEngine.
  3. Powered by Groq ('openai/gpt-oss-20b') and local sentence-transformers.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine, TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    # PART 1: INITIALIZE MODELS (Groq + Local Embeddings)
    print_banner("PART 1: INITIALIZING GROQ & EMBEDDINGS")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY missing from .env!")
    
    Settings.llm = Groq(model="openai/gpt-oss-20b", api_key=groq_api_key, temperature=0.1)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Groq ('openai/gpt-oss-20b') and Hugging Face Embeddings ready.")

    # PART 2: LOAD SEPARATE DOCUMENTS & BUILD ISOLATED INDICES
    print_banner("PART 2: ISOLATED DOCUMENT INDICES")
    data_dir = Path(__file__).parent.parent / "09_llamaindex_ingestion" / "data"

    # Ingest Compliance Policy
    compliance_docs = SimpleDirectoryReader(input_files=[str(data_dir / "compliance_policy.md")]).load_data()
    compliance_index = VectorStoreIndex.from_documents(compliance_docs)
    print("Compliance Index built.")

    # Ingest System Architecture
    architecture_docs = SimpleDirectoryReader(input_files=[str(data_dir / "system_architecture.txt")]).load_data()
    architecture_index = VectorStoreIndex.from_documents(architecture_docs)
    print("Architecture Index built.")

    # PART 3: SUB-QUESTION QUERY ENGINE (QUERY DECOMPOSITION)
    print_banner("PART 3: SUB-QUESTION DECOMPOSITION PIPELINE")

    compliance_tool = QueryEngineTool(
        query_engine=compliance_index.as_query_engine(),
        metadata=ToolMetadata(
            name="compliance_engine",
            description="Contains enterprise compliance policies, encryption standards, and RPO/RTO disaster recovery rules."
        )
    )

    architecture_tool = QueryEngineTool(
        query_engine=architecture_index.as_query_engine(),
        metadata=ToolMetadata(
            name="architecture_engine",
            description="Contains Core Banking System (CBS) architecture, branch scales, Kafka event streams, and database specs."
        )
    )

    # Use LLMQuestionGenerator so it works reliably with any open-source model:
    question_gen = LLMQuestionGenerator.from_defaults(llm=Settings.llm)

    # SubQuestionQueryEngine breaks complex questions down across the tools
    sub_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[compliance_tool, architecture_tool],
        question_gen=question_gen,
        verbose=True # Watch the generated sub-questions print in console!
    )

    multi_doc_query = (
        "Compare our Disaster Recovery RPO/RTO requirements in the compliance policy "
        "with how the Core Banking System (CBS) guarantees data integrity."
    )
    print(f"\nTarget Multi-Hop Query:\n\"{multi_doc_query}\"\n")
    
    sub_response = sub_engine.query(multi_doc_query)
    print("\n--- Groq Synthesized Comparative Answer ---")
    print(sub_response.response)
    print("-" * 70)

    # PART 4: HyDE (HYPOTHETICAL DOCUMENT EMBEDDINGS)
    print_banner("PART 4: HyDE QUERY TRANSFORMATION")

    # HyDEQueryTransform generates a hypothetical passage to match vocabulary
    hyde_transform = HyDEQueryTransform(include_original=True)

    base_query_engine = architecture_index.as_query_engine(similarity_top_k=1)
    
    # Wrap with TransformQueryEngine
    hyde_query_engine = TransformQueryEngine(
        query_engine=base_query_engine,
        query_transform=hyde_transform
    )

    # An abstract question with zero keyword overlap:
    abstract_query = "How do we prevent financial double-spending anomalies?"
    print(f"Abstract User Query: \"{abstract_query}\"")
    print("(Notice: 'double-spending' does not appear in simple questions, but HyDE bridges it!)\n")
    hyde_response = hyde_query_engine.query(abstract_query)
    print("--- HyDE Final Answer ---")
    print(hyde_response.response)
    print("-" * 70)
    print(f"Source Node Matched: {hyde_response.source_nodes[0].metadata.get('file_name')}")

if __name__ == "__main__":
    main()
