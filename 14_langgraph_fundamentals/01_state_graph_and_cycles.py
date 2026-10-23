"""
01_state_graph_and_cycles.py
======================================================================
LangGraph Fundamentals: State Graphs, Message Reducers, and Cyclic Loops
======================================================================
Demonstrates:
  1. Defining a shared AgentState with the add_messages reducer.
  2. Building the Agent Node and Tool Node.
  3. Implementing Conditional Edges for dynamic loop routing.
  4. Closing the cycle: tools -> agent.
  5. Multi-step autonomous tool execution (Account balance + Currency conversion).
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

# PART 1: DYNAMIC TOOLS DEFINITION
class AccountBalanceInput(BaseModel):
    account_id: str = Field(description="Bank account ID formatted as 'ACC-XXXX'.")

@tool("get_account_balance", args_schema=AccountBalanceInput)
def get_account_balance(account_id: str) -> str:
    """Retrieves the current cleared balance and base currency for an account."""
    mock_balances = {
        "ACC-1001": {"balance": 15400.50, "currency": "USD"},
        "ACC-2002": {"balance": 4200.00, "currency": "EUR"},
        "ACC-3003": {"balance": 890000.00, "currency": "JPY"},
    }
    acc = mock_balances.get(account_id.upper())
    if not acc:
        return f"Error: Account '{account_id}' was not found in core ledger."
    return f"Account {account_id.upper()}: Cleared Balance = {acc['balance']} {acc['currency']}"

class CurrencyConvertInput(BaseModel):
    amount: float = Field(description="The numeric amount to convert.")
    from_currency: str = Field(description="3-letter source currency code (e.g. USD, EUR, GBP).")
    to_currency: str = Field(description="3-letter target currency code (e.g. USD, EUR, GBP).")

@tool("convert_currency", args_schema=CurrencyConvertInput)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts a monetary amount between currencies using spot exchange rates."""
    exchange_rates = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
        ("USD", "GBP"): 0.79,
        ("GBP", "USD"): 1.27,
        ("USD", "JPY"): 155.0,
    }
    pair = (from_currency.upper(), to_currency.upper())
    rate = exchange_rates.get(pair)
    if not rate:
        return f"Unsupported currency pair: {from_currency} -> {to_currency}"
    converted = round(amount * rate, 2)
    return f"{amount} {from_currency.upper()} = {converted} {to_currency.upper()} (Spot Rate: {rate})"

# PART 2: STATE SCHEMA & GRAPH NODES
class AgentState(TypedDict):
    """
    The central memory of our graph.
    Annotated with 'add_messages' so every node return appends to the history
    instead of replacing the existing messages list.
    """
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize Groq Model and bind our tools
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY missing from environment or .env file!")

llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=groq_api_key, temperature=0.1)
tools = [get_account_balance, convert_currency]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """
    Agent Node: Inspects the conversation history and decides whether
    to respond directly or invoke one or more tools.
    """
    print("\n  [AGENT NODE] Reasoning over conversation state...")
    response = llm_with_tools.invoke(state["messages"])
    # Return updates to state['messages'] (reduced via add_messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", END]:
    """
    Conditional Edge Router:
    Inspects the last message from the Agent Node.
    - If tool_calls exist -> Route to the 'tools' node.
    - If no tool_calls exist -> Route to END (conversation complete).
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"  [ROUTER] Detected {len(last_message.tool_calls)} tool call(s) -> Routing to 'tools' node.")
        return "tools"
    print("  [ROUTER] No tool calls detected -> Routing to END.")
    return END

# PART 3: ASSEMBLING AND COMPILING THE CYCLIC STATE GRAPH
def build_agent_graph():
    builder = StateGraph(AgentState)

    # 1. Add Nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))  # ToolNode handles automatic tool execution & ToolMessage generation!
    
    # 2. Add Entry Edge
    builder.add_edge(START, "agent")
    
    # 3. Add Conditional Edge: agent -> (tools OR END)
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    # 4. Add the Cyclic Edge: tools -> agent (CRITICAL!)
    # Once tools finish executing, control loops back to the agent
    # so the agent can read the tool output and formulate the final response.
    builder.add_edge("tools", "agent")

    # 5. Compile into an executable Runnable
    return builder.compile()

# PART 4: TESTING MULTI-STEP AGENTIC EXECUTION
def main():
    app = build_agent_graph()

    # TEST 1: Direct Answer (Zero cycles)
    print_banner("TEST 1: DIRECT QUERY (NO TOOLS NEEDED)")
    query_1 = "Explain the difference between fiat currency and central bank digital currency (CBDC) in one sentence."
    print(f"User Query: \"{query_1}\"")
    
    events = app.stream({"messages": [HumanMessage(content=query_1)]}, stream_mode="values")
    for event in events:
        last_msg = event["messages"][-1]
    print(f"\nFinal Response:\n{last_msg.content}")

    # TEST 2: Multi-Step Reasoning Cycle (Balance Lookup + Currency Conversion)
    print_banner("TEST 2: MULTI-STEP AGENTIC CYCLE (2 TOOLS IN SEQUENCE)")
    query_2 = "What is the balance of account ACC-1001, and how much is that in EUR?"
    print(f"User Query: \"{query_2}\"")

    # Step through every state change in the graph
    print("\n--- Streaming Graph Node Executions ---")
    step_num = 1
    for chunk in app.stream({"messages": [HumanMessage(content=query_2)]}):
        for node_name, node_update in chunk.items():
            print(f"\n[Step {step_num}] Node '{node_name}' executed.")
            latest_message = node_update["messages"][-1]
            if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                for tc in latest_message.tool_calls:
                    print(f"  * Requested Tool: {tc['name']}({tc['args']})")
            elif hasattr(latest_message, "content") and latest_message.content:
                print(f"  * Message Content: {latest_message.content[:150]}...")
            step_num += 1
    
    # Fetch final complete state
    final_state = app.invoke({"messages": [HumanMessage(content=query_2)]})
    print("\n--- Final Synthesized Response ---")
    print(final_state["messages"][-1].content)

if __name__ == "__main__":
    main()