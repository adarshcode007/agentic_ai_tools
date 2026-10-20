"""
01_hub_and_serverless_inference.py
Phase 1: Hugging Face Ecosystem & Local Model Deployment
Subtopic: Hugging Face Hub, Model Management, Cache Inspection & Serverless Inference
Key Learning Concepts:
1. Hugging Face Authentication:
   - Managing access tokens (HF_TOKEN) for gated models (Llama 3, Mistral).
2. Local Cache Hygiene (scan_cache_dir):
   - Inspecting downloaded weights on disk and calculating total disk consumption.
3. Serverless Inference API (InferenceClient):
   - Executing remote cloud inference on open-source foundation models with zero local GPU compute.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, InferenceClient, scan_cache_dir

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # PART 1: AUTHENTICATION & IDENTITY (HfApi)
    print_banner("PART 1: HUGGING FACE AUTHENTICATION & IDENTITY")

    api = HfApi(token=hf_token)

    if hf_token:
        try:
            user_info = api.whoami()
            print(f"[AUTHENTICATED] Logged in as: {user_info.get('name', 'Unknown')}")
            print(f"  * Account Type : {user_info.get('type', 'user')}")
            print(f"  * Can access gated models: YES")
        except Exception as e:
            print(f"[WARNING] Invalid HF_TOKEN provided: {e}")
    else:
        print("[ANONYMOUS MODE] No 'HF_TOKEN' found in .env.")
        print("  -> You can still download public models and inspect your local cache.")
        print("  -> (Tip: Add HF_TOKEN=hf_... to .env to access gated models like Llama 3).")
    

    # PART 2: LOCAL DISK CACHE INSPECTION (scan_cache_dir)
    print_banner("PART 2: LOCAL DISK CACHE INSPECTION")

    try:
        cache_info = scan_cache_dir()
        total_size_mb = cache_info.size_on_disk / (1024 * 1024)
        print(f"Hugging Face Cache Directory: {cache_info.size_on_disk_str}")
        print(f"Total Repositories Downloaded: {len(cache_info.repos)}")
        print(f"Total Disk Consumed           : {total_size_mb:.2f} MB\n")
        print(f"{'Repository ID':<45} | {'Type':<8} | {'Size on Disk'}")
        print("-" * 70)
        for repo in cache_info.repos:
            print(f"{repo.repo_id:<45} | {repo.repo_type:<8} | {repo.size_on_disk_str}")
        
        print("\n💡 Architectural Insight:")
        print("  Hugging Face stores deduplicated model snapshots and git revisions.")
        print("  In production Docker containers or CI/CD pipelines, engineers use")
        print("  'scan_cache_dir()' or 'huggingface-cli delete-cache' to prevent disk exhaustion.")
    except Exception as e:
        print(f"Error scanning local cache: {e}")

    # PART 3: SERVERLESS INFERENCE API (InferenceClient)
    print_banner("PART 3: SERVERLESS CLOUD INFERENCE (InferenceClient)")

    target_model = "Qwen/Qwen2.5-72B-Instruct"
    print(f"Target Serverless Model: '{target_model}'")
    print("Connecting to Hugging Face Cloud Infrastructure (Zero Local GPU required)...")

    # Initialize client (works with HF_TOKEN or anonymously for public endpoints)
    client = InferenceClient(token=hf_token)

    prompt = (
        "Explain in two concise sentences why vector quantization "
        "is critical for running LLMs on edge devices."
    )
    print(f"\nUser Prompt: \"{prompt}\"")

    try:
        # Request chat completion from cloud-hosted model
        response = client.chat_completion(
            model=target_model,
            messages=[{"role":"user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()
        print("\nCloud Model Completion:")
        print("-" * 70)
        print(f"\"{reply}\"")
        print("-" * 70)
    
    except Exception as e:
        print(f"\n[NOTE on Serverless API]: {e}")
        print("  (Serverless endpoints require a valid free HF_TOKEN in .env).")
        print("  You can generate a free token at: https://huggingface.co/settings/tokens")
    
    # PART 4: GATED MODELS & ENTERPRISE COMPLIANCE
    print_banner("PART 4: GATED MODELS & PRODUCTION BEST PRACTICES")
    print("1. What are Gated Models?")
    print("   * Models like Meta-Llama-3, Mistral, and Google Gemma are open-weights,")
    print("     but require users to accept a license agreement on their HF webpage.")
    print("   * If you try to download 'meta-llama/Meta-Llama-3-8B' without an authenticated")
    print("     token that has accepted the license, Hugging Face returns HTTP 403 Forbidden.")
    print("\n2. Enterprise Docker Best Practice:")
    print("   * Never run 'huggingface-cli login' interactively inside Docker.")
    print("   * Set the environment variable: ENV HF_TOKEN=hf_... inside your Dockerfile/K8s.")
    print("   * Or mount a shared volume to /root/.cache/huggingface/ to reuse cached weights.")
    print("=" * 70)

if __name__ == "__main__":
    main()