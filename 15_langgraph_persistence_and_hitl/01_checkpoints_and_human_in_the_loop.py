"""
01_checkpoints_and_human_in_the_loop.py
======================================================================
LangGraph Persistence & Human-in-the-Loop (HITL) Approval Gates
======================================================================
Demonstrates:
  1. Using MemorySaver to persist multi-turn conversation state across turns.
  2. Multi-tenant session isolation using thread_id.
  3. Interrupt Gates (interrupt_before=["tools"]) to pause execution before critical tools.
  4. State inspection and resuming execution (app.invoke(None)) after human approval.
  5. Time-travel inspection using app.get_state_history().
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

# PART 1: DEFINE LOW-RISK AND HIGH-RISK FINANCIAL TOOLS
class ReadBalanceInput(BaseModel):
    account_id: str = Field(description="Account ID to check.")

@tool("check_balance", args_schema=ReadBalanceInput)
def check_balance(account_id: str) -> str:
    """Safe read-only operation: returns available balance."""
    return f"Account {account_id.upper()}: Available Balance = $45,000.00 USD."

class WireTransferInput(BaseModel):
    recipient_account: str = Field(description="Target account formatted as 'ACC-XXXX'.")
    amount: float = Field(description="Monetary amount in USD to wire.")
    reason: str = Field(description="Business justification for the transfer.")

@tool("execute_wire_transfer", args_schema=WireTransferInput)
def execute_wire_transfer(recipient_account: str, amount: float, reason: str) -> str:
    """CRITICAL / HIGH-STAKES: Irreversibly wires funds to an external account."""
    return f"TRANSFER SUCCESSFUL: Wires ${amount:,.2f} USD to {recipient_account.upper()}. Reference TX-99018. Reason: {reason}"


# PART 2: STATE SCHEMA & GRAPH ASSEMBLY
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize LLM with tools
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY missing from environment or .env file!")

llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=groq_api_key, temperature=0.1)
tools = [check_balance, execute_wire_transfer]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = (
    "You are an enterprise banking AI assistant. You have access to tools: 'check_balance' and 'execute_wire_transfer'. "
    "When a user asks to check a balance or transfer/wire funds, you MUST call the corresponding tool. "
    "Do not respond conversationally when an action is requested."
)

def agent_node(state: AgentState):
    """Agent Node: Decides whether to answer or call a tool."""
    print("  [AGENT NODE] Reasoning over conversation state...")
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", END]:
    """Router: Checks for pending tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def build_persistent_hitl_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    # 1. Instantiate the Checkpointer
    checkpointer = MemorySaver()

    # 2. Compile with BOTH the checkpointer AND interrupt_before
    # interrupt_before=["tools"] pauses graph execution right before the tool node runs!
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"]
    )


# PART 3: DEMONSTRATING MULTI-TURN PERSISTENT MEMORY

def test_multi_turn_memory(app):
    print_banner("DEMO 1: MULTI-TURN MEMORY ACROSS TURNS VIA THREAD_ID")

    # Session 1 for user Alice
    alice_config = {"configurable": {"thread_id": "session-alice-101"}}

    # Turn 1: Alice introduces herself and asks to check her balance
    print("\n--- Turn 1: Alice shares her account and checks balance ---")
    query_turn1 = "Hi! My name is Alice. Please check my balance for account ACC-8812."
    print(f"Alice: \"{query_turn1}\"")

    # Run the graph. (Notice: check_balance will hit the interrupt gate, so we auto-resume it)
    app.invoke({"messages": [HumanMessage(content=query_turn1)]}, config=alice_config)
    # Resume past the read-only tool
    state_turn1 = app.invoke(None, config=alice_config)
    print(f"Agent Response:\n{state_turn1['messages'][-1].content}")

    # Turn 2: Alice asks a follow-up without repeating her name or account!
    print("\n--- Turn 2: Alice asks follow-up (Zero redundant context) ---")
    query_turn2 = "What was my name and what account did I just ask about?"
    print(f"Alice: \"{query_turn2}\"")

    state_turn2 = app.invoke({"messages": [HumanMessage(content=query_turn2)]}, config=alice_config)
    print(f"Agent Response:\n{state_turn2['messages'][-1].content}")


# PART 4: DEMONSTRATING HUMAN-IN-THE-LOOP APPROVAL FOR WIRE TRANSFER
def test_human_in_the_loop_approval(app):
    print_banner("DEMO 2: HUMAN-IN-THE-LOOP (HITL) APPROVAL GATE")

    transfer_config = {"configurable": {"thread_id": "session-wire-transfer-99"}}
    
    # User initiates a critical transfer of $25,000
    wire_query = "Please wire $25,000 to account ACC-4401 for quarterly server infrastructure invoices."
    print(f"User Request: \"{wire_query}\"")

    # 1. Run the graph until the interrupt gate
    print("\n[Executing graph up to the Interrupt Gate...]")
    app.invoke({"messages": [HumanMessage(content=wire_query)]}, config=transfer_config)
    
    # 2. Inspect the current paused state
    current_snapshot = app.get_state(transfer_config)
    print(f"\n⏸️ GRAPH IS PAUSED!")
    print(f"Next Node Scheduled to Run: {current_snapshot.next}")
    
    # Inspect the pending tool calls inside the last message
    pending_ai_message = current_snapshot.values["messages"][-1]

    # Defensive check: ensure the model actually called a tool!
    if "tools" not in current_snapshot.next or not getattr(pending_ai_message, "tool_calls", None):
        print("\n[Notice] Model responded directly with text instead of a toll call:")
        print(pending_ai_message.content)
        return
    
    pending_tool_call = pending_ai_message.tool_calls[0]

    print("\n" + "#" * 60)
    print("  HUMAN COMPLIANCE VERIFICATION REQUIRED")
    print("#" * 60)
    print(f"  Tool Name : {pending_tool_call['name']}")
    print(f"  Recipient : {pending_tool_call['args']['recipient_account']}")
    print(f"  Amount    : ${pending_tool_call['args']['amount']:,.2f} USD")
    print(f"  Reason    : {pending_tool_call['args']['reason']}")
    print("#" * 60)

    # 3. Simulate Human Decision
    human_approval = "YES"  # Simulate human clicking 'Approve' in a web UI
    print(f"\nCompliance Officer Decision: {human_approval}")

    if human_approval == "YES":
        print("\n[Approved! Resuming execution from checkpoint with app.invoke(None)...]")
        # Passing None tells LangGraph: "Resume from the exact frozen state, do not inject new messages"
        final_state = app.invoke(None, config=transfer_config)
        print(f"\nFinal Agent Output:\n{final_state['messages'][-1].content}")
    else:
        print("\n[Rejected! Transfer aborted.]")

    # 4. Time-Travel History Inspection
    print_banner("DEMO 3: TIME-TRAVEL AUDIT LOG (STATE HISTORY)")
    print("Inspecting all saved checkpoints for this thread:")
    history = list(app.get_state_history(transfer_config))
    for i, state_step in enumerate(history):
        checkpoint_id = state_step.config["configurable"].get("checkpoint_id", "initial")
        node_visited = state_step.metadata.get("source", "start")
        print(f"  Checkpoint {i+1} | Node: {node_visited} | Checkpoint ID: {checkpoint_id[:8]}... | Messages: {len(state_step.values.get('messages', []))}")
    
def main():
    app = build_persistent_hitl_graph()
    test_multi_turn_memory(app)
    test_human_in_the_loop_approval(app)

if __name__ == "__main__":
    main()

