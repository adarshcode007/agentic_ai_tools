"""
01_supervisor_worker_pattern.py
======================================================================
Multi-Agent Orchestration in LangGraph: The Supervisor-Worker Pattern
======================================================================
Demonstrates:
  1. Defining specialized, domain-specific worker tools.
  2. Building isolated worker nodes (Market Researcher & Compliance Auditor).
  3. Building a central Supervisor Node with Pydantic structured output routing.
  4. Cyclic multi-agent handoffs: Supervisor -> Worker -> Supervisor -> END.
  5. Multi-step collaborative reasoning across distinct specialists.
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

# PART 1: SPECIALIZED WORKER TOOLS
# --- RESEARCHER TOOLS ---
class MarketSearchInput(BaseModel):
    ticker: str = Field(description="Asset symbol or ticker (e.g., 'BTC', 'ETH', 'NVDA').")

@tool("search_market_data", args_schema=MarketSearchInput)
def search_market_data(ticker: str) -> str:
    """Fetches real-time price, volatility, and market cap for an asset."""
    mock_db = {
        "BTC": {"price": "$92,450 USD", "24h_volatility": "3.8%", "risk_category": "HIGH"},
        "ETH": {"price": "$3,420 USD", "24h_volatility": "4.2%", "risk_category": "HIGH"},
        "NVDA": {"price": "$142.50 USD", "24h_volatility": "2.1%", "risk_category": "MEDIUM"},
        "US_TREASURY": {"yield": "4.35%", "24h_volatility": "0.1%", "risk_category": "VERY_LOW"},
    }
    data = mock_db.get(ticker.upper())
    if not data:
        return f"Market data unavailable for ticker: '{ticker}'."
    return f"Market Data for {ticker.upper()}: Price={data.get('price', 'N/A')}, Volatility={data['24h_volatility']}, Risk Category={data['risk_category']}"


# --- COMPLIANCE AUDITOR TOOLS ---
class RegulatoryAuditInput(BaseModel):
    asset_class: str = Field(description="Asset class: 'CRYPTO', 'EQUITY', or 'TREASURY'.")
    allocation_amount: float = Field(description="Total capital amount in USD to allocate.")

@tool("audit_regulatory_compliance", args_schema=RegulatoryAuditInput)
def audit_regulatory_compliance(asset_class: str, allocation_amount: float) -> str:
    """Audits whether a proposed investment allocation complies with institutional banking limits."""
    # Institutional exposure limits
    limits = {
        "CRYPTO": 50_000.0,      # Max $50k in digital assets
        "EQUITY": 250_000.0,     # Max $250k in equities
        "TREASURY": 1_000_000.0  # Max $1M in sovereign debt
    }
    cap = limits.get(asset_class.upper(), 25_000.0)
    if allocation_amount > cap:
        return f"VIOLATION: Allocation of ${allocation_amount:,.2f} exceeds institutional cap of ${cap:,.2f} for {asset_class.upper()}. Action: REJECT or SCALE DOWN."
    return f"COMPLIANT: Allocation of ${allocation_amount:,.2f} is within institutional cap of ${cap:,.2f} for {asset_class.upper()}. Risk audit PASSED."

# PART 2: MULTI-AGENT STATE SCHEMA & SPECIALIZED WORKERS
class MultiAgentState(TypedDict):
    """Shared state across the supervisor and all worker agents."""
    messages: Annotated[list[BaseMessage], add_messages]
    next_worker: str

# Initialized Groq Model 
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY missing from environment or .env file")

llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=groq_api_key, temperature=0.1)

# --- HELPER: Worker Node Execution Function ---
def create_worker_node(worker_name: str, worker_system_prompt: str, worker_tools: list):
    """Creates a self-contained agent worker node with its own private tools."""
    worker_llm = llm.bind_tools(worker_tools)
    tool_map = {t.name: t for t in worker_tools}

    def worker_node(state: MultiAgentState):
        print(f"\n  [{worker_name.upper()} WORKER] Activated. Analyzing task...")

        # Prepend specialized worker instructions
        messages = [SystemMessage(content=worker_system_prompt)] + state["messages"]
        response = worker_llm.invoke(messages)

        # If the worker called tools, execute them immediately
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_results = []
            for tc in response.tool_calls:
                selected_tool = tool_map[tc["name"]]
                print(f"  [{worker_name.upper()} TOOL] Running {tc['name']}({tc['args']})")
                res = selected_tool.invoke(tc["args"])
                tool_results.append(str(res))
            
            # Formulate final worker synthesis with tool results
            follow_up = messages + [response] + [
                AIMessage(content=f"Tool Execution Output:\n" + "\n".join(tool_results))
            ]
            final_worker_msg = worker_llm.invoke(follow_up)
            content = f"[{worker_name.capitalize()} Report]:\n{final_worker_msg.content}"
        else:
            content = f"[{worker_name.capitalize()} Report]:\n{response.content}"
        return {"messages": [AIMessage(content=content, name=worker_name)]}
    
    return worker_node


# Instantiate Specialized Workers
researcher_node = create_worker_node(
    worker_name="researcher",
    worker_system_prompt=(
        "You are a specialized Market Research Analyst. Use your tools to fetch market data, "
        "asset pricing, and risk metrics. Summarize facts concisely. Do not audit regulatory limits."
    ),
    worker_tools=[search_market_data]
)
auditor_node = create_worker_node(
    worker_name="auditor",
    worker_system_prompt=(
        "You are an Institutional Compliance & Risk Auditor. Inspect proposed allocations "
        "and determine whether they pass banking regulations using your audit tools. Be strict and precise."
    ),
    worker_tools=[audit_regulatory_compliance]
)


# PART 3: THE SUPERVISOR ROUTER NODE & GRAPH ASSEMBLY
class SupervisorDecision(BaseModel):
    """The structured routing decision made by the supervisor."""
    next_worker: Literal["researcher", "auditor", "FINISH"] = Field(
        description="The next specialized worker to delegate to, or 'FINISH' if the request has been fully answered."
    )
    reasoning: str = Field(
        description="Brief justification for why this worker was chosen or why the task is finished."
    )

SUPERVISOR_SYSTEM_PROMPT = """You are the Lead Financial Operations Supervisor managing a team of two specialists:
1. 'researcher': Researches market prices, volatility, and asset metrics (search_market_data).
2. 'auditor': Audits investment capital limits, compliance caps, and regulatory constraints (audit_regulatory_compliance).
Rules:
- Read the entire conversation history carefully.
- If market data or asset price is needed and not yet provided -> delegate to 'researcher'.
- If compliance, allocation caps, or regulations need verification -> delegate to 'auditor'.
- Once all required information has been gathered and reported -> output 'FINISH'.
"""

def supervisor_node(state: MultiAgentState):
    """Supervisor Node: Decides which specialized worker should act next."""
    print("\n  [SUPERVISOR] Evaluating team progress and deciding next delegation...")
    
    supervisor_llm = llm.with_structured_output(SupervisorDecision)
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
    decision: SupervisorDecision = supervisor_llm.invoke(messages)
    
    print(f"  [SUPERVISOR DECISION] Delegate to -> '{decision.next_worker}' | Reason: {decision.reasoning}")
    return {"next_worker": decision.next_worker}

def build_supervisor_graph():
    builder = StateGraph(MultiAgentState)

    # 1. Add all Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("auditor", auditor_node)

    # 2. Entrypoint is always the Supervisor
    builder.add_edge(START, "supervisor")

    # 3. Dynamic Conditional Edge: Supervisor delegates to worker OR finishes
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next_worker"],
        {
            "researcher": "researcher",
            "auditor": "auditor",
            "FINISH": END
        }
    )

    # 4. Critical Cyclic Handoff: Workers ALWAYS report back to the Supervisor!
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("auditor", "supervisor")
    return builder.compile()


# PART 4: TESTING MULTI-AGENT COLLABORATION SCENARIOS
def main():
    app = build_supervisor_graph()

    # SCENARIO 1: Single-Specialist Query (Market Research only)
    print_banner("SCENARIO 1: RESEARCH DELEGATION")
    query_1 = "What is the current market price and volatility for Bitcoin (BTC)?"
    print(f"User Request: \"{query_1}\"")

    for chunk in app.stream({"messages": [HumanMessage(content=query_1)]}):
        for node_name in chunk.keys():
            print(f"  --> Step completed by node: '{node_name}'")

    # SCENARIO 2: Multi-Specialist Collaboration (Research + Compliance Audit)
    print_banner("SCENARIO 2: MULTI-AGENT COLLABORATION (RESEARCH -> AUDIT -> FINISH)")
    query_2 = (
        "We want to invest $75,000 into Bitcoin (BTC). "
        "First, fetch the current market data for BTC. "
        "Then, audit whether a $75,000 CRYPTO allocation complies with institutional banking limits."
    )
    print(f"User Request: \"{query_2}\"")

    final_state = app.invoke({"messages": [HumanMessage(content=query_2)]})

    print("\n" + "=" * 70)
    print("  FINAL COLLABORATIVE EXECUTIVE REPORT")
    print("=" * 70)
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"\n{msg.content}\n" + "-" * 50)
    
if __name__ == "__main__":
    main()
