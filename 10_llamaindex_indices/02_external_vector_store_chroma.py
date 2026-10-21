"""
02_external_vector_store_chroma.py
======================================================================
Connecting LlamaIndex to Production Vector Stores (ChromaDB)
======================================================================
Demonstrates:
  1. Instantiating a persistent ChromaDB client and collection.
  2. Wrapping the collection with LlamaIndex's ChromaVectorStore.
  3. Passing the custom vector_store into StorageContext.
  4. Building VectorStoreIndex directly on top of ChromaDB.
  5. Proving that vectors & metadata exist natively in ChromaDB.
"""

from pathlib import Path
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import MockLLM

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    print_banner("CONNECTING LLAMAINDEX TO CHROMADB")

    # 1. Configure local embeddings
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = MockLLM()

    # 2. Initialize ChromaDB Persistent Client
    chroma_dir = Path(__file__).parent / "chroma_storage"
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    # Create or get collection
    collection_name = "enterprise_knowledge_base"
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    print(f"ChromaDB persistent collection '{collection_name}' ready at: {chroma_dir}")

    # 3. Wrap Chroma collection in LlamaIndex's ChromaVectorStore adapter
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 4. Inject the vector_store into LlamaIndex's StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Ingest documents and build VectorStoreIndex
    data_dir = Path(__file__).parent.parent / "09_llamaindex_ingestion" / "data"
    documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()

    print(f"\nIndexing {len(documents)} document(s) directly into ChromaDB...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    print("VectorStoreIndex successfully synced with ChromaDB!")

    # 6. Verify directly in ChromaDB (outside LlamaIndex!)
    print("\n--- Direct ChromaDB Inspection ---")
    print(f"Total Vectors in Chroma Collection: {chroma_collection.count()}")
    peek = chroma_collection.peek(limit=1)
    print(f"Sample Chroma ID: {peek['ids'][0]}")
    print(f"Sample Chroma Metadata Keys: {list(peek['metadatas'][0].keys())}")

    # 7. Execute Query via LlamaIndex Retriever backed by Chroma
    print_banner("RETRIEVING VIA CHROMADB HNSW BACKEND")
    retriever = index.as_retriever(similarity_top_k=1)
    results = retriever.retrieve("What is the Recovery Point Objective?")

    print(f"Query Result:")
    print(f"  * Matched File : {results[0].node.metadata.get('file_name')}")
    print(f"  * Top Score    : {results[0].score:.4f}")
    print(f"  * Text Snippet : {results[0].node.text[:90].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    main()