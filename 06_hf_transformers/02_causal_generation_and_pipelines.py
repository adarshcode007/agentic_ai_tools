"""
02_causal_generation_and_pipelines.py
Phase 1: Hugging Face Ecosystem & Local Model Deployment
Subtopic: Causal Language Modeling, Batch Generation with Left-Padding & Decoding Strategies
Key Learning Concepts:
1. Low-Level Generation (AutoModelForCausalLM):
   - Loading local decoder weights.
   - Strictly enforcing left-padding: tokenizer.padding_side = "left".
2. Decoding Strategies:
   - Greedy Search (do_sample=False): Deterministic argmax.
   - Temperature Scaling: Softmax entropy manipulation (z / T).
   - Top-K & Top-P (Nucleus Sampling): Dynamic candidate truncation.
3. High-Level Hugging Face Pipeline:
   - pipeline("text-generation"): Encapsulating tokenization, inference, and decoding.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    # Set seed for reproducing sampling experiments
    torch.manual_seed(42)

    # PART 1: LOAD CASUAL MODEL & CONFIGURE LEFT-PADDING
    print_banner("PART 1: LOADING LOCAL CASUAL MODEL (GPT-2)")

    model_name = "gpt2"
    print(f"Loading Tokenizer: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # CRITICAL INVARIANT: For causal batch generation, set padding_side to LEFT
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer configured with: padding_side='{tokenizer.padding_side}'")
    print(f"Loading Causal Weights: '{model_name}' (124M parameters)...")

    # Load model in evaluation mode (inference only)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully into local RAM!")

    # PART 2: LOW-LEVEL BATCH GENERATION WITH LEFT-PADDING
    print_banner("PART 2: BATCH GENERATION WITH LEFT-PADDING")

    batch_prompts = [
        "Artificial Intelligence will",
        "The quickest path to mastering distributed systems and cloud architecture is",
    ]

    print("Batch Prompts (Unequal Lengths):")
    for i, p in enumerate(batch_prompts, 1):
        print(f"  [{i}] \"{p}\"")
    
    # Tokenize as PyTorch tensors (pt) with Left-Padding
    inputs = tokenizer(
        batch_prompts,
        padding=True,
        return_tensors="pt"
    )

    print("\nLeft-Padded 'input_ids' Matrix (Notice pad tokens on the LEFT of Prompt 1):")
    print(inputs["input_ids"])

    print("\n'attention_mask' Matrix (0 for left pad tokens, 1 for real words):")
    print(inputs["attention_mask"])

    # Generate completions in parallel
    print("\nGenerating completions with model.generate()...")
    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=25,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  # Greedy search for baseline comparison
        )
    
    print("\nGenerated Completions (Greedy Search):")
    print("-" * 70)
    for i, seq in enumerate(generated_tokens, 1):
        completion = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"Prompt [{i}] Continuation:\n  \"{completion}\"\n")
    
    # PART 3: DECODING STRATEGIES (Greedy vs. Temperature vs. Top-P/Top-K)
    print_banner("PART 3: DECODING STRATEGIES COMPARISON")

    test_prompt = "In the future, autonomous AI agents will"
    prompt_input = tokenizer(test_prompt, return_tensors="pt")

    # 1. Greedy Search (do_sample=False): Deterministic argmax
    with torch.no_grad():
        greedy_ids = model.generate(
            **prompt_input,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    greedy_text = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)

    # 2. Temperature Sampling (High vs Low Entropy)
    with torch.no_grad():
        # Low Temperature (T=0.3): Conservative, focused
        low_temp_ids = model.generate(
            **prompt_input,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3
        )
        # High Temperature (T=1.4): Creative, high variance
        high_temp_ids = model.generate(
            **prompt_input,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=1.4
        )
    low_temp_text = tokenizer.decode(low_temp_ids[0], skip_special_tokens=True)
    high_temp_text = tokenizer.decode(high_temp_ids[0], skip_special_tokens=True)

    # 3. Top-P (Nucleus) + Top-K Sampling: The Industry Gold Standard
    with torch.no_grad():
        topp_ids = model.generate(
            **prompt_input,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,      # Only consider top 50 candidates
            top_p=0.90     # Cumulative probability cutoff at 90%
        )
    topp_text = tokenizer.decode(topp_ids[0], skip_special_tokens=True)

    print(f"Base Prompt: \"{test_prompt}\"\n")
    print(f"[1] Greedy Search (Deterministic):\n    \"{greedy_text}\"\n")
    print(f"[2] Low Temp (T=0.3 - Focused):\n    \"{low_temp_text}\"\n")
    print(f"[3] High Temp (T=1.4 - High Entropy / Creative):\n    \"{high_temp_text}\"\n")
    print(f"[4] Top-K (50) + Top-P (0.90) Sampling (Standard Production Balanced):\n    \"{topp_text}\"\n")

    # PART 4: HIGH-LEVEL PIPELINE ABSTRACTION
    print_banner("PART 4: HIGH-LEVEL HUGGING FACE PIPELINE ABSTRACTION")

    # Create end-to-end generator pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cpu"
    )

    print("Running generator pipeline on custom prompt...")
    pipeline_result = generator(
        "To build a resilient vector database,",
        max_new_tokens=25,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )

    print("\nPipeline Output:")
    print(f"  \"{pipeline_result[0]['generated_text']}\"")
    print("\n" + "=" * 70)
    print("  CAUSAL GENERATION & PIPELINES DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()