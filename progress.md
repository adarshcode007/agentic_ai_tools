# Applied AI / RAG Frameworks & Tools Career Roadmap & Progress Tracker

This document maps out the essential industry frameworks, vector databases, and ecosystem tools required to check every box on enterprise AI engineering job applications. 

Having already mastered the fundamentals **from first principles (scratch)** in Phase 0–5, this roadmap fast-tracks translating that foundational depth into the industry's most in-demand frameworks: **LangChain / LangGraph, LlamaIndex, Pinecone, Qdrant, Chroma, FAISS, Hugging Face, and Modern LLM APIs**.

---

## 📍 Current Status: `Phase 0 Complete ✅ ──▶ Phase 1 Complete ✅ ──▶ Phase 2 Complete ✅ ──▶ Phase 3 (LangChain & LangGraph — State Machines & Multi-Agent Orchestration) 🟡` — Ready to Start 🚀

```
[✅ Phase 0] ──▶ [✅ Phase 1] ──▶ [✅ Phase 2] ──▶ [🟡 Phase 3] ──▶ [⚪ Phase 4] ──▶ [⚪ Phase 5]
Vector Stores     Hugging Face     LlamaIndex       LangGraph        Provider APIs    Production
(FAISS/Qdrant/    (Transformers/   (RAG/Query       (Agents/State    (OpenAI/Claude/  Capstone
 Pinecone/Chroma)  Embeddings)      Engines)         Machines)        LiteLLM)         Portfolio
```

---

## 🗺️ Detailed Syllabus & Roadmap

---

### Phase 0: Vector Databases & Similarity Search Engines (COMPLETED ✅)
*Mastering local embedded indices and production cloud vector databases, their indexing algorithms, and metadata filtering.*

- [x] **Local Vector Search with FAISS (Facebook AI Similarity Search)**:
  - Building exact (`IndexFlatL2`, `IndexFlatIP`) vs approximate nearest neighbor indices (`IndexHNSWFlat`, `IndexIVFFlat`).
  - Understanding clustering, Voronoi cells, and quantization (Product Quantization - PQ) for memory reduction.
- [x] **Embedded Prototyping with ChromaDB**:
  - Setting up Chroma collections with persistent SQLite/DuckDB backing.
  - Adding documents, custom metadata, and executing filtered vector queries (`where={"category": "finance"}`).
- [x] **Production Cloud Vector Store with Pinecone**:      
  - Creating Serverless / Pod indexes with dimensional configuration and metric selection (cosine vs dotproduct).
  - Multi-tenant data segregation using Pinecone **Namespaces**.
  - Hybrid search using dense embeddings + sparse vectors (SPLADE / BM25).
  - Metadata pre-filtering under high cardinality.
- [x] **High-Performance Self-Hosted Vector Engine with Qdrant**:
  - Running Qdrant locally via Docker / Python client.
  - Single-stage filtered HNSW search with payload schemas (pre-filtering without index degradation).
  - Snapshotting, collection aliases, and point payloads.

---

### Phase 1: Hugging Face Ecosystem & Local Model Deployment (COMPLETED ✅)
*Moving beyond proprietary APIs: running open-source embeddings, cross-encoders, and quantized local models.*

- [x] **Embedding & Re-ranking with `sentence-transformers`**:
  - Benchmarking open-source embedding models (e.g., `BAAI/bge-large-en-v1.5`, `nomic-embed-text`).
  - Implementing local cross-encoder re-ranking pipelines (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on real GPUs / CPUs.
- [x] **Hugging Face `transformers` & Tokenizers**:
  - Tokenizer mechanics: subword splitting (BPE/WordPiece), token IDs, attention masks, truncation strategies.
  - Loading open-source models (Llama 3 / Mistral / Qwen / GPT-2) with `AutoModelForCausalLM` and `pipeline`.
- [x] **Quantization & Local Inference**:
  - Running 4-bit and 8-bit quantized models using `bitsandbytes`, AWQ, or GGUF (via `llama-cpp-python`).
  - Offloading inference between CPU, RAM, and GPU VRAM.
- [x] **Hugging Face Hub & Model Management**:
  - Authenticating with HF tokens, local cache architecture inspection (`scan_cache_dir`), gated model compliance, and serverless Inference Endpoints / Inference Providers router.

---

### Phase 2: LlamaIndex — The Production RAG Standard (COMPLETED ✅)
*Mastering the data framework designed specifically for advanced indexing, document intelligence, and retrieval.*

- [x] **Data Ingestion & Document Abstraction**:
  - Ingesting unstructured data using `SimpleDirectoryReader` and file-specific parsers (PDF, Markdown, HTML).
  - Document vs `TextNode` abstractions: chunking, node relationships (`NodeRelationship.PARENT`, `CHILD`).
- [x] **Index Architectures**:
  - `VectorStoreIndex`: Creating, persisting, and querying dense vector indices.
  - `SummaryIndex` and `KeywordTableIndex`: When and how to pair them with vector stores.
  - Connecting LlamaIndex to external storage backends (Qdrant, Pinecone, Chroma).
- [x] **Advanced Retrieval & Query Engines**:
  - `RouterQueryEngine`: Dynamically routing queries between a summary engine and a vector search engine.
  - `SubQuestionQueryEngine`: Decomposing multi-step questions into sub-queries across different documents.
  - HyDE (Hypothetical Document Embeddings) Query Transform (`HyDEQueryTransform`).
  - Node Postprocessors: Injecting `SentenceTransformerRerank` and similarity cutoffs.
- [x] **LlamaIndex Workflows & Agentic RAG**:
  - Event-driven RAG workflows using the new LlamaIndex Workflows engine.

---

### Phase 3: LangChain & LangGraph — State Machines & Multi-Agent Orchestration (IN PROGRESS 🟡)
*Mastering the industry standard for stateful, cyclic multi-agent graphs and enterprise orchestration.*

- [x] **Modern LangChain Core & LCEL (LangChain Expression Language)**:
  - Building pipelines with `ChatPromptTemplate | ChatModel | OutputParser`.
  - Streaming, batching, and async execution (`ainvoke`, `astream`, `abatch`).
  - LangChain dynamic tool definitions using `@tool` decorator with Pydantic schemas.
- [ ] **LangGraph Fundamentals**:
  - Why LangGraph replaces rigid DAG chains: cyclic execution loops, conditional routing, and state persistence.
  - Defining typed schemas with `TypedDict` and `Annotated` reducers.
  - Building nodes (workers), edges, and conditional edges (`add_conditional_edges`).
- [ ] **Stateful Persistence & Human-in-the-Loop (HITL)**:
  - Checkpointing states with `MemorySaver` / SQLite checkpointers to support time travel and rollback.
  - Implementing interrupt gates (`interrupt()`) for human approval before critical tool execution.
- [ ] **Multi-Agent Systems with LangGraph**:
  - Rebuilding the Supervisor-Worker and Swarm handoff patterns using LangGraph subgraphs.

---

### Phase 4: Enterprise LLM Provider APIs & Model Portability (NOT STARTED ⚪)
*Mastering production API parameters, enterprise safety, and vendor-neutral routing.*

- [ ] **Advanced OpenAI API Capabilities**:
  - Strict structured outputs with `response_format={"type": "json_schema", ...}`.
  - Parallel function calling and streaming tool chunks.
  - OpenAI Assistants API, Threads, Runs, and code interpreter tool.
- [ ] **Vendor-Neutral Model Gateway (LiteLLM)**:
  - Unified interface across 100+ LLMs (OpenAI, Anthropic Claude, Groq, Mistral, Ollama).
  - Fallback logic: Auto-routing from Anthropic/OpenAI to Groq on rate limits (`HTTP 429`) or timeouts.
  - Cost tracking and budget caps per user/session.

---

### Phase 5: Production Capstone — Enterprise Agentic RAG Platform (NOT STARTED ⚪)
*A flagship portfolio project that connects all framework tools into a cohesive product to present in interviews.*

- [ ] **Project Architecture: "Enterprise Multi-Source Knowledge & Action Copilot"**:
  - **Data Layer (LlamaIndex + Pinecone/Qdrant)**: Ingests technical documentation, contracts, and APIs with parent-child chunks and metadata pre-filtering.
  - **Model Layer (Hugging Face + LiteLLM)**: Local Cross-Encoder re-ranker combined with dynamic LLM routing (OpenAI / Claude / Groq).
  - **Orchestration Layer (LangGraph)**: Multi-agent supervisor workflow with human-in-the-loop approvals for external tool actions.
  - **Production Polish**: Dockerized deployment, FastAPI streaming endpoints, and end-to-end evaluation suite.

---

## 💡 How First-Principles Knowledge Maps to Frameworks

| Built from Scratch in `Phase 0-5` | Modern Framework Equivalent |
| :--- | :--- |
| `while` loop with tool calling in `agent.py` | LangChain `AgentExecutor` / LangGraph `StateGraph` |
| SQLite session history & rolling compaction | LangChain `ChatMessageHistory` / LangGraph Checkpointers |
| Recursive parent-child splitting in `rag.py` | LlamaIndex `HierarchicalNodeParser` / LangChain `ParentDocumentRetriever` |
| BM25 + Vector cosine + RRF in `rag.py` | Pinecone / Qdrant Hybrid Search & LlamaIndex `RouterQueryEngine` |
| Cross-Encoder reranker in `rag.py` | Hugging Face `sentence-transformers` & LlamaIndex `SentenceTransformerRerank` |
| Telemetry traces table in `db.py` | Langfuse / LangSmith / OpenTelemetry observability integration |
