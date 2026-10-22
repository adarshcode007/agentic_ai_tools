"""
01_event_driven_agentic_rag.py
======================================================================
LlamaIndex Workflows: Event-Driven Agentic Corrective RAG (CRAG)
======================================================================
Demonstrates:
  1. Defining custom typed Event classes (RetrieveEvent, RerankEvent, SynthesizeEvent).
  2. Building an asynchronous Workflow subclass with @step handlers.
  3. Corrective loop: Evaluating retrieval quality and triggering query rewrites.
  4. Cross-Encoder re-ranking and Groq LLM synthesis culminating in StopEvent.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Event
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# 1. DEFINE TYPED WORKFLOW EVENTS
class RerankEvent(Event):
    """Fired when retrieved; succeeds and nodes need cross-encoder re-ranking."""
    query: str 
    nodes: list[NodeWithScore]

class RewriteQueryEvent(Event):
    """Fired when retrieval quality is below threshold, trigerring query expansion."""
    original_query: str

class SynthesizeEvent(Event):
    """Fired when nodes have been filtered and re-ranked, ready for LLM synthesis."""
    query: str
    nodes: list[NodeWithScore]

# 2. DEFINE THE EVENT-DRIVEN AGENTIC WORKFLOW
class CorrectiveRAGWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, reranker: SentenceTransformerRerank, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.reranker = reranker
        self.retriever = index.as_retriever(similarity_top_k=4)
    
    @step
    async def retrieve(self, ev: StartEvent) -> RerankEvent | RewriteQueryEvent:
        """Step 1: Retrieve candidate nodes and evaluate retrieval quality."""
        query = ev.get("query")
        print(f"\n[STEP 1: RETRIEVE] Searching index for: \"{query}\"")
        
        nodes = self.retriever.retrieve(query)
        if not nodes:
            print("  * Warning: No nodes retrieved! Triggering query rewrite...")
            return RewriteQueryEvent(original_query=query)
        top_score = nodes[0].score if nodes[0].score is not None else 0.0
        print(f"  * Retrieved {len(nodes)} candidate node(s). Top Vector Score: {top_score:.4f}")

        # Quality Check: If the match is too weak (< 0.20), trigger self-correction
        if top_score < 0.20:
            print("  * Low confidence retrieval (< 0.20). Triggering query rewrite loop...")
            return RewriteQueryEvent(original_query=query)
        return RerankEvent(query=query, nodes=nodes)
    
    @step
    async def rewrite_query(self, ev: RewriteQueryEvent) -> StartEvent:
        """Step 2: Self-correction loop. Reformulate query to improve recall."""
        print(f"\n[STEP 2: SELF-CORRECTION] Rewriting ambiguous query: \"{ev.original_query}\"")
        prompt = (
            f"The following query had poor search retrieval: '{ev.original_query}'. "
            "Rephrase it into a technical, clear search query for an enterprise banking & compliance system. "
            "Output ONLY the rephrased query string."
        )
        response = await Settings.llm.acomplete(prompt)
        new_query = response.text.strip().replace('"', '')
        print(f"  * Rewritten Query: \"{new_query}\"")
        
        # Loop back to start with the improved query!
        return StartEvent(query=new_query)
    
    @step
    async def rerank(self, ev: RerankEvent) -> SynthesizeEvent:
        """Step 3: Run local Cross-Encoder re-ranking over candidate nodes."""
        print(f"\n[STEP 3: RERANK] Scoring {len(ev.nodes)} node(s) with Cross-Encoder...")
        
        # Re-score and pick top 2
        reranked_nodes = self.reranker.postprocess_nodes(ev.nodes, query_str=ev.query)
        for i, n in enumerate(reranked_nodes):
            print(f"  * Rank {i+1}: {n.metadata.get('file_name')} (Cross-Encoder Logit: {n.score:.4f})")
            
        return SynthesizeEvent(query=ev.query, nodes=reranked_nodes)
    
    @step
    async def synthesize(self, ev: SynthesizeEvent) -> StopEvent:
        """Step 4: Groq LLM synthesizes final answer with source grounding."""
        print(f"\n[STEP 4: SYNTHESIZE] Generating final answer with Groq...")
        
        context_str = "\n\n".join([f"Source ({n.metadata.get('file_name')}):\n{n.node.text}" for n in ev.nodes])
        prompt = (
            f"Context Information:\n{context_str}\n\n"
            f"User Question: {ev.query}\n"
            "Provide a concise, grounded technical answer based strictly on the context above."
        )
        
        response = await Settings.llm.acomplete(prompt)
    
        # Return StopEvent to end the workflow and return the final payload
        return StopEvent(result={
                "query": ev.query,
                "answer": response.text.strip(),
                "sources": [n.metadata.get("file_name") for n in ev.nodes]
            })
    
# 3. MAIN EXECUTION
def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

async def main():
    print_banner("INTITIALIZING EVENT-DRIVEN AGENTIC WORKFLOW")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY missing from .env!")
    
    Settings.llm = Groq(model="openai/gpt-oss-20b", api_key=groq_api_key, temperature=0.1)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    data_dir = Path(__file__).parent.parent / "09_llamaindex_ingestion" / "data"
    documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()
    index = VectorStoreIndex.from_documents(documents)
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=2
    )

    workflow = CorrectiveRAGWorkflow(index=index, reranker=reranker, timeout=60.0)

    # TEST 1: Clear, direct  query (Clean 1-pass execution)
    print_banner("EXECUTION RUN 1: DIRECT QUERY")
    result_1 = await workflow.run(query="What is the RPO and RTO for our financial ledger?")
    print("\n--- Final Workflow Result ---")
    print(result_1["answer"])
    print(f"Sources: {result_1['sources']}")

    # TEST 2: Vague query (Triggers the self-correcting rewrite loop!)
    print_banner("EXECUTION RUN 2: AMBIGUOUS QUERY (SELF-CORRECTION LOOP)")
    result_2 = await workflow.run(query="Tell me about machines and storage stuff.")
    print("\n--- Final Workflow Result ---")
    print(result_2["answer"])
    print(f"Sources: {result_2['sources']}")

if __name__ == "__main__":
    asyncio.run(main())
