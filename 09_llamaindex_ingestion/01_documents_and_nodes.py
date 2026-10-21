"""
01_documents_and_nodes.py
======================================================================
LlamaIndex Core Ingestion, Document Abstractions, and Node Topologies
======================================================================
Demonstrates:
  1. Directory-level ingestion using SimpleDirectoryReader.
  2. Manual Document instantiation with fine-grained metadata masking
     (excluded_llm_metadata_keys vs. excluded_embed_metadata_keys).
  3. Chunking Documents into atomic TextNode instances using SentenceSplitter.
  4. Inspecting NodeRelationship pointers (SOURCE, PREVIOUS, NEXT).
"""

from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import TextNode, NodeRelationship
from llama_index.core.node_parser import SentenceSplitter

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    # PART 1: DIRECTORY INGESTION WITH SimpleDirectoryReader
    print_banner("PART 1: INGESTION WITH SimpleDirectoryReader")
    data_dir = Path(__file__).parent / "data"

    reader = SimpleDirectoryReader(input_dir=str(data_dir))
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents(s) from '{data_dir.name}':\n")
    for doc in documents:
        print(f"Doc ID   : {doc.doc_id}")
        print(f"File Name: {doc.metadata.get('file_name')}")
        print(f"File Size: {doc.metadata.get('file_size')} bytes")
        print(f"Snippet  : {doc.text[:80].replace(chr(10), ' ')}...")
        print("-" * 50)

    # PART 2: MANUAL DOCUMENT CREATION & METADATA MASKING
    print_banner("PART 2: MANUAL DOCUMENT & METADATA MASKING")

    # Enterprise scenario: We have an internal engineering runbook containing
    # sensitive database credentials and tenant routing tags
    custom_doc = Document(
        text=(
            "PostgreSQL Connection Pool Settings: Max connections = 100. "
            "Idle timeout = 300s. Connection string uses SSL verify-full. "
            "Failover replica is located in us-east-1b."
        ),
        metadata={
            "service": "payments-backend",
            "environment": "production",
            "db_secret_arn": "arn:aws:secretsmanager:us-east-1:123456789:secret:pg_cred",
            "compliance_tier": "PCI-DSS-Level-1"
        },
        # Configure metadata masking:
        # Exclude 'db_secret_arn' from being passed to the LLM prompt!
        excluded_llm_metadata_keys=["db_secret_arn"],
        # Exclude 'db_secret_arn' and 'environment' from being embedded in the vector index!
        excluded_embed_metadata_keys=["db_secret_arn", "environment"]
    )

    print("--- [1] Raw Underlying Content (metadata_mode='all') ---")
    print(custom_doc.get_content(metadata_mode="all"))

    print("\n--- [2] Prompt Content Fed to LLM (metadata_mode='llm') ---")
    # 'db_secret_arn' is automatically stripped to prevent prompt leakage!
    print(custom_doc.get_content(metadata_mode="llm"))

    print("\n--- [3] Content Embedded by Vector Model (metadata_mode='embed') ---")
    # 'db_secret_arn' AND 'environment' are stripped so they don't pollute the embedding vector!
    print(custom_doc.get_content(metadata_mode="embed"))

    # PART 3: CHUNKING DOCUMENTS INTO TextNode INSTANCES
    print_banner("PART 3: CHUNKING INTO TextNode OBJECTS")

    # SentenceSplitter respects sentence boundaries while enforcing chunk size
    splitter = SentenceSplitter(chunk_size=120, chunk_overlap=30)

    # Split both custom_doc and the directory documents into nodes
    all_docs = [custom_doc] + documents
    nodes = splitter.get_nodes_from_documents(all_docs)
    print(f"Generated {len(nodes)} TextNode(s) from {len(all_docs)} Document(s).")

    # PART 4: GRAPH TOPOLOGY & NODE RELATIONSHIP
    print_banner("PART 4: NODE GRAPH RELATIONSHIPS (SOURCE, PREV, NEXT)")
    
    for i, node in enumerate(nodes):
      print(f"\n[Node {i}] ID: {node.node_id}")
      print(f"Text Snippet: {node.text[:65].replace(chr(10),' ')}...")

      # 1. SOURCE: Points back to parent Document ID
      source_rel = node.relationships.get(NodeRelationship.SOURCE)
      parent_id = source_rel.node_id if source_rel else "None"
      print(f"   * Parent Doc (SOURCE)   : {parent_id}")

      # 2. PREVIOUS: Pointer to preceding chunk (doubly linked list)
      prev_rel = node.relationships.get(NodeRelationship.PREVIOUS)
      prev_id = prev_rel.node_id if prev_rel else "None (Start of Document)"
      print(f"   * Previous Node (PREV) : {prev_id}")

      # 3. NEXT: Pointer to succeeding chunk
      next_rel = node.relationships.get(NodeRelationship.NEXT)
      next_id = next_rel.node_id if next_rel else "None (End of Document)"
      print(f"   * Next Node (NEXT)     : {next_id}")

      # Notice how metadata cascaded from the parent document to this Node!
      print(f"   * Inherited Metadata    : {list(node.metadata.keys())}")

if __name__ == "__main__":
    main()




    