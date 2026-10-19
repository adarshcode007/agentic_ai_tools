"""
01_tokenizer_mechanics.py
Phase 1: Hugging Face Ecosystem & Local Model Deployment
Subtopic: Tokenizer Mechanics, Subword Splitting, Attention Masks & Padding
Key Learning Concepts:
1. Subword Algorithms:
   - Byte-Pair Encoding (BPE - GPT/Llama) vs. WordPiece (BERT).
   - Why LLMs fail at character counting ("strawberry"), string reversal, and arithmetic.
2. Token Whitespace Sensitivity:
   - Why " Hello" and "Hello" are completely different token IDs.
3. Batch Padding & Attention Masks:
   - Why unequal sequence lengths require pad tokens.
   - Why attention_mask (1s vs 0s) is mathematically mandatory in the Softmax layer.
4. Special Tokens:
   - [CLS], [SEP], <s>, </s>, <|endoftext|>.
"""

from transformers import AutoTokenizer

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    # PART 1: LOAD INDUSTRY TOKENIZERS (BPE vs. WordPiece)
    print_banner("PART 1: LOADING TOKENIZERS (BPE vs. WordPiece)")

    # 1. GPT-2 Tokenizer (Byte-Pair Encoding - BPE, standard for generative LLMs)
    print("Loading GPT-2 Tokenizer (BPE)...")
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"  * GPT-2 Vocab Size : {gpt_tokenizer.vocab_size:,} tokens")

    # 2. BERT Tokenizer (WordPiece, standard for encoders/classifiers)
    print("Loading BERT Tokenizer (WordPiece)...")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"  * BERT Vocab Size  : {bert_tokenizer.vocab_size:,} tokens")


    # PART 2: SUBWORD SPLITTING & THE "STRAWBERRY" LLM BLIND SPOT
    print_banner("PART 2: SUBWORD SPLITTING & THE LLM BLIND SPOT")

    test_words = [
        "strawberry",
        "UserAuthenticationError_v2",
        "unconstitutional",
    ]

    for word in test_words:
        print(f"\nAnalyzing Word: '{word}'")
        # GPT-2 BPE Tokens
        gpt_tokens = gpt_tokenizer.tokenize(word)
        gpt_ids = gpt_tokenizer.encode(word)
        print(f"  [GPT-2 BPE]     Tokens: {gpt_tokens}")
        print(f"                  IDs   : {gpt_ids}")

        # BERT WordPiece Tokens (WordPiece uses '##' for subwords)
        bert_tokens = bert_tokenizer.tokenize(word)
        bert_ids = bert_tokenizer.encode(word, add_special_tokens=False)
        print(f"  [BERT WordPiece] Tokens: {bert_tokens}")
        print(f"                  IDs   : {bert_ids}")
    
    print("\n💡 Architectural Insight:")
    print("  Notice 'strawberry' -> ['straw', 'berry'] (or ['str', 'aw', 'berry']).")
    print("  The LLM NEVER sees the letters 's-t-r-a-w-b-e-r-r-y'. It only receives two")
    print("  integer numbers. Asking an LLM 'how many r's in strawberry' is asking it to")
    print("  guess internal character spelling from compressed vector concepts!")

    # PART 3: WHITESPACE & CAPITALIZATION SENSITIVITY
    print_banner("PART 3: WHITESPACE SENSITIVITY (Why Leading Spaces Matter)")

    variants = ["Hello", " Hello", " hello", "HELLO"]

    print(f"{'Input String':<15} | {'Tokens':<20} | {'Token IDs'}")
    print("-" * 55)
    for v in variants:
        tokens = gpt_tokenizer.tokenize(v)
        ids = gpt_tokenizer.encode(v)
        print(f"{repr(v):<15} | {str(tokens):<20} | {ids}")

    print("\n💡 Architectural Insight:")
    print("  Notice that 'Hello' and ' Hello' (with leading space) have DIFFERENT token IDs!")
    print("  In prompt engineering, an accidental space before a stop token or variable")
    print("  changes the integer fed to the neural network entirely.")

    # PART 4: BATCH PADDING, TRUNCATION & THE ATTENTION MASK
    print_banner("PART 4: BATCH PADDING, TRUNCATION & THE ATTENTION MASKS")

    # GPT-2 has no default pad token, so we set it to EOS token
    if gpt_tokenizer.pad_token is None:
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    
    batch_sentences = [
        "Deep learning is transformative.",
        "Transformers revolutionized natural language processing and computer vision.",
    ]

    print("Raw Batch Sentences:")
    for i, s in enumerate(batch_sentences, 1):
        print(f"  Sentence {i}: \"{s}\"")
    
    # Encode with dynamic padding and max_length truncation
    encoded_batch = gpt_tokenizer(
        batch_sentences,
        padding=True,          # Pad shorter sentences to match the longest
        truncation=True,       # Truncate if exceeds max_length
        max_length=16,
        return_tensors="np",   # Return as NumPy arrays for inspection
    )

    print("\nEncoded Batch Structure:")
    print(f"  Keys in output dictionary: {list(encoded_batch.keys())}")

    print("\n1. 'input_ids' Matrix (Shape: {}):".format(encoded_batch["input_ids"].shape))
    print(encoded_batch["input_ids"])

    print("\n2. 'attention_mask' Matrix (Shape: {}):".format(encoded_batch["attention_mask"].shape))
    print(encoded_batch["attention_mask"])

    print("\n💡 Architectural Insight on Attention Masks:")
    print("  Why are attention masks mandatory?")
    print("  The 1s indicate REAL words; the 0s indicate PAD tokens.")
    print("  In the Transformer Attention equation Softmax(Q * K^T / sqrt(d)),")
    print("  mask 0 forces the attention score to -Infinity.")
    print("  exp(-Inf) = 0.0, which guarantees that padding words do NOT contribute")
    print("  to the attention weights of real words!")

    # PART 5: DECODING (Converting IDs Back to Text)
    print_banner("PART 5: DETOKENIZATION & SPECIAL TOKENS")

    first_seq_ids = encoded_batch["input_ids"][0]

    # Decode with special tokens (shows the pad/eos tokens)
    decoded_with_special = gpt_tokenizer.decode(first_seq_ids, skip_special_tokens=False)
    # Decode skipping special tokens (clean human text)
    decoded_clean = gpt_tokenizer.decode(first_seq_ids, skip_special_tokens=True)

    print(f"Original IDs: {list(first_seq_ids)}")
    print(f"With Special Tokens : \"{decoded_with_special}\"")
    print(f"Clean (Skipped)     : \"{decoded_clean}\"")
    print("\n" + "=" * 70)
    print("  TOKENIZER MECHANICS DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()

