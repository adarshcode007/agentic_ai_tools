"""
01_lcel_runnables_and_tools.py
======================================================================
Modern LangChain Core & LCEL (LangChain Expression Language)
======================================================================
Demonstrates:
  1. The Runnable Protocol: Composing chains with the pipe operator (|).
  2. Execution modes: .invoke(), .stream(), and .batch().
  3. Async execution: .ainvoke() and .astream().
  4. Type-safe Dynamic Tools using the @tool decorator and Pydantic schemas.
  5. Tool Binding (.bind_tools) and parsing tool_calls from AIMessage.
  6. End-to-end tool execution and response synthesis with ToolMessage.
"""

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

# PART 1: DYNAMIC TOOLS DEFINITION (PYDANTIC SCHEMAS + @tool)
# In modern LangChain, tools are defined using the @tool decorator.
# Providing an explicit Pydantic args_schema gives strict type safety
# and generates the exact JSON schema required by Groq/OpenAI function calling. 

class VATCalculationInput(BaseModel):
    amount: float = Field(description="The base monetary amount before tax in EUR or USD.")
    country_code: str = Field(description="Two-letter ISO country code (e.g., 'DE', 'FR', 'GB', 'US').")

@tool("calculate_vat", args_schema=VATCalculationInput) 
def calculate_vat(amount: float, country_code: str) -> str:
    """Calculates Value Added Tax (VAT) and the total cost for a given country."""
    vat_rates = {
        "DE": 0.19,  # Germany 19%
        "FR": 0.20,  # France 20%
        "GB": 0.20,  # United Kingdom 20%
        "US": 0.08,  # United States (avg sales tax ~8%)
    }
    rate = vat_rates.get(country_code.upper(), 0.15)  # Default fallback 15%
    tax = round(amount * rate, 2)
    total = round(amount + tax, 2)
    return f"Country: {country_code.upper()} | Base: {amount} | Tax Rate: {rate*100:.0f}% | Tax: {tax} | Total: {total}"

class LedgerAuditInput(BaseModel):
    account_id: str = Field(description="Bank account ID formatted as 'ACC-XXXX'.")
    limit: int = Field(default=3, description="Number of historical ledger entries to inspect.")

@tool("audit_bank_ledger", args_schema=LedgerAuditInput)
def audit_bank_ledger(account_id: str, limit: int = 3) -> str:
    """Audits recent ledger transactions for double-spending anomalies and compliance."""
    # Simulated ledger database records
    mock_transactions = [
        {"tx_id": "TX-101", "type": "DEPOSIT", "amount": 10000.0, "status": "CONFIRMED"},
        {"tx_id": "TX-102", "type": "TRANSFER", "amount": -1500.0, "status": "CONFIRMED"},
        {"tx_id": "TX-103", "type": "WITHDRAWAL", "amount": -200.0, "status": "CONFIRMED"},
    ]
    records = mock_transactions[:limit]
    return f"Audit for {account_id}: Inspected {len(records)} transaction(s). Integrity: PASSED (Zero double-spend anomalies)."


async def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY missing from environment or .env file")
    
    # 1. Initialize ChatGroq Model
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.1
    )

    # PART 2: THE RUNNABLE PROTOCOL & LCEL CHAIN PIPELINE
    print_banner("PART 1: THE RUNNABLE PROTOCOL (PROMPT | MODEL | PARSER)")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise financial AI assistant. Give concise, technical responses."),
        ("human", "{question}")
    ])
    parser = StrOutputParser()

    # Composing the LCEL chain using the bitwise OR (|) operator:
    # prompt produces PromptValue -> model consumes it and produces AIMessage -> parser extracts string
    chain = prompt | llm | parser

    # A) Synchronous .invoke()
    print("\n[1. Testing .invoke()]")
    response_invoke = chain.invoke({"question": "Explain what a Two-Phase Commit protocol is in one sentence."})
    print(f"Result:\n{response_invoke.strip()}")

    # B) Token Streaming with .stream()
    print("\n[2. Testing .stream() - Token-by-Token Streaming]")
    print("Streamed: ", end="", flush=True)
    for chunk in chain.stream({"question": "List 3 common security threats to distributed financial ledgers."}):
        print(chunk, end="", flush=True)
    print()

    # C) Batch Processing with .batch() (Parallel Execution)
    print("\n[3. Testing .batch() - Concurrent Execution]")
    batch_inputs = [
        {"question": "What does RPO stand for in disaster recovery?"},
        {"question": "What does RTO stand for in disaster recovery?"}
    ]
    batch_results = chain.batch(batch_inputs)
    for q, res in zip(batch_inputs, batch_results):
        print(f"  * Q: {q['question']}")
        print(f"    A: {res.strip()}\n")
    
    # D) Async Execution with .ainvoke()
    print("[4. Testing .ainvoke() - Asynchronous Call]")
    async_res = await chain.ainvoke({"question": "What is idempotent API design in one phrase?"})
    print(f"Async Result: {async_res.strip()}")

    # PART 3: TOOL BINDING & FUNCTION CALLING INSPECTION
    print_banner("PART 2: TOOL BINDING & FUNCTION CALLING (.bind_tools)")

    # Register our tools with the LLM
    tools = [calculate_vat, audit_bank_ledger]
    tool_map = {t.name: t for t in tools}

    # .bind_tools converts our Python tools into JSON Schemas and binds them to the model
    llm_with_tools = llm.bind_tools(tools)

    user_query = "Please calculate the VAT for a purchase of 4500 EUR in Germany (DE)."
    print(f"\nUser Query: \"{user_query}\"")

    # When invoked, the LLM decides whether to return text or request a tool execution
    ai_msg: AIMessage = llm_with_tools.invoke(user_query)
    
    print("\n[Inspecting Model Output]")
    print(f"Model Content Text: '{ai_msg.content}'")
    print(f"Detected Tool Calls Count: {len(ai_msg.tool_calls)}")
    
    for i, tc in enumerate(ai_msg.tool_calls):
        print(f"\n  --- Tool Call {i+1} ---")
        print(f"  Name: {tc['name']}")
        print(f"  Arguments: {tc['args']}")
        print(f"  Tool Call ID: {tc['id']}")
    
    # PART 4: COMPLETE TOOL EXECUTION & SYNTHESIS LOOP
    print_banner("PART 3: TOOL EXECUTION & CONVERSATION SYNTHESIS")
    
    # Start the message history with the user's initial prompt and model's tool call request
    messages = [
        HumanMessage(content=user_query),
        ai_msg
    ]

    # Execute the requested tools
    for tc in ai_msg.tool_calls:
        tool_to_call = tool_map[tc["name"]]
        print(f"\n[Executing Tool '{tc['name']}' with args: {tc['args']}]")

        # Invoke the tool with the arguments generated by the model
        tool_output = tool_to_call.invoke(tc["args"])
        print(f"  Raw Tool Output: {tool_output}")

        # Wrap the result in a ToolMessage containing the matching tool_call_id
        messages.append(ToolMessage(
            tool_call_id=tc["id"],
            content=str(tool_output)
        ))

    # Send the full history (Human -> AI tool_call -> ToolMessage result) back to the LLM
    print("\n[Synthesizing Final Response with Model...]")
    final_response = llm_with_tools.invoke(messages)
    print("\nFinal Synthesized Answer:")
    print(final_response.content)

    # PART 5: MULTI-TOOL CALL SCENARIO (Ledge Audit)
    print_banner("PART 4: SECOND TOOL TEST (LEDGER AUDIT)")

    audit_query = "Audit the ledger transactions for account ACC-9021. Check the last 2 records."
    print(f"\nUser Query: \"{audit_query}\"")
    
    audit_ai_msg = llm_with_tools.invoke(audit_query)
    print(f"Generated Tool Calls: {audit_ai_msg.tool_calls}")
    
    audit_messages = [HumanMessage(content=audit_query), audit_ai_msg]
    for tc in audit_ai_msg.tool_calls:
        selected_tool = tool_map[tc["name"]]
        res = selected_tool.invoke(tc["args"])
        audit_messages.append(ToolMessage(tool_call_id=tc["id"], content=str(res)))
    
    final_audit = llm_with_tools.invoke(audit_messages)
    print(f"\nFinal Audit Report:\n{final_audit.content}")

if __name__ == "__main__":
    asyncio.run(main())