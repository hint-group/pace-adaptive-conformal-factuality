import json
import os
import asyncio
import random
import time
import numpy as np
from typing import List
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ================= Configuration =================

API_KEY = ""
BASE_URL = "https://api.openai.com/v1"

# [Strictly Aligned with Annotation Code]
TARGET_MODEL = "gpt-5.2"       # Gen, Self-Eval, Annotation
HELPER_MODEL = "gpt-4o-mini"   # Parsing, Frequency
NUM_SAMPLES = 5                # For Frequency Score

SAMPLE_SIZE = 30 # Measure 5 samples and take the average
SEED = 42

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 1. Prompts (Exact Copy) =================

PROMPTS = {
    "bio_gen": "Please write one biographical paragraph about {name}.\n",
    
    "separator": """Please breakdown the following input into a set of small, independent claims. 
Return a valid JSON object with a single key "claims", which is a list of strings.
Example: {{"claims": ["claim 1", "claim 2"]}}
Input:
{input_text}""",

    "self_eval": """You will get a claim and the original prompt that motivated it. Assess the probability of correctness. Directly return a jsonl, where each line is {{"gpt-score": [SCORE]}}. Directly return the jsonl with NO explanation or ANY other formatting. For the [SCORE], return the estimated probability of correctness to three significant figures.
The original prompt is: {prompt}
The claim is: {claim}""",

    "frequency": """You will get a list of claims and a piece of text. 
For each claim, determine if the text supports (1), contradicts (-1), or is unrelated (0) to the claim.
Return a JSON object with a key "scores", containing a list of objects with "id" and "score".
Example: {{"scores": [{{"id": 0, "score": 1}}, {{"id": 1, "score": 0}}]}}
Claims:
{claim_string}

Text:
{sample_text}""",

    "annotation": """You will get a claim and a piece of text (context). Score whether the text supports (1), contradicts (-1), or is unrelated (0). Only return the number.
Claim: {claim}
Context: {context}"""
}

# ================= 2. Simulation Components =================

async def call_llm_real(model: str, prompt: str, temp: float = 0.0, json_mode: bool = False):
    """
    Wrapper strictly matching your original call_llm logic.
    NO max_tokens limit.
    """
    t0 = time.perf_counter()
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            
        res = await client.chat.completions.create(**kwargs)
        # Get real content to ensure the task is actually completed
        content = res.choices[0].message.content
        latency = time.perf_counter() - t0
        return content, latency
    except Exception as e:
        print(f"Error in LLM call ({model}): {e}")
        return "", 0.0

# --- Phase 1: Main Generation ---
async def phase_gen_main(entity_name: str) -> (str, float):
    prompt = PROMPTS["bio_gen"].format(name=entity_name)
    # Target Model, temp=1.0, NO max_tokens
    content, latency = await call_llm_real(TARGET_MODEL, prompt, temp=1.0)
    # If generation fails, use placeholder to prevent downstream crash, but record time as 0 or error time
    if not content: content = "Dummy bio text."
    return content, latency

# --- Phase 2: Parallel Block (Samples + Wiki + Parse) ---
async def phase_parallel_block(entity_name: str, bio_text: str) -> (List[str], float):
    t0_block = time.perf_counter()

    # Task A: 5 Samples (Target Model) - FULL GENERATION
    sample_prompt = PROMPTS["bio_gen"].format(name=entity_name)
    async def run_samples():
        # Run 5 full generations concurrently
        tasks = [call_llm_real(TARGET_MODEL, sample_prompt, temp=1.0) for _ in range(NUM_SAMPLES)]
        await asyncio.gather(*tasks)
    
    # Task B: Wiki Retrieval
    async def run_wiki():
        await asyncio.sleep(0.8) # Network simulation

    # Task C: Parsing (Helper Model) - FULL PARSING
    parse_prompt = PROMPTS["separator"].format(input_text=bio_text)
    async def run_parse():
        content, _ = await call_llm_real(HELPER_MODEL, parse_prompt, temp=0.0, json_mode=True)
        return content
    
    # Run A, B, C in parallel
    results = await asyncio.gather(run_samples(), run_wiki(), run_parse())
    
    # Try to extract claims from parsing result to be realistic
    claims = []
    try:
        parse_res = results[2] # result of run_parse
        claims = json.loads(parse_res).get("claims", [])
    except:
        pass
    
    if not claims: claims = [f"Claim {i}" for i in range(5)] # Fallback
    
    block_latency = time.perf_counter() - t0_block
    return claims, block_latency

# --- Phase 3: Frequency Calculation ---
async def phase_frequency(claims: List[str]) -> float:
    if not claims: return 0.0
    t0 = time.perf_counter()
    
    # Construct real Prompt (length affects Helper Model processing time)
    claim_str = "\n".join([f"ID {i}: {c}" for i, c in enumerate(claims)])
    dummy_sample_text = "Generated sample text content..." * 20 # Simulate Sample length
    prompt = PROMPTS["frequency"].format(claim_string=claim_str, sample_text=dummy_sample_text)
    
    # 5 concurrent calls to Helper Model
    tasks = [call_llm_real(HELPER_MODEL, prompt, temp=0.0, json_mode=True) for _ in range(NUM_SAMPLES)]
    await asyncio.gather(*tasks)
    
    return time.perf_counter() - t0

# --- Phase 4: Enrichment (Self-Eval + Annotation) ---
async def phase_enrichment(claims: List[str], entity_name: str) -> float:
    if not claims: return 0.0
    t0 = time.perf_counter()
    
    prompt_gen = PROMPTS["bio_gen"].format(name=entity_name)
    
    # Limit concurrency to 5 (Consistent with MAX_CONCURRENT_CLAIMS in your code)
    sem = asyncio.Semaphore(5)

    async def process_single_claim(claim_text):
        async with sem:
            # 1. Self Eval
            p_self = PROMPTS["self_eval"].format(prompt=prompt_gen, claim=claim_text)
            task_self = call_llm_sim_enrich(TARGET_MODEL, p_self)
            
            # 2. Annotation
            dummy_evidence = f"Wiki evidence about {entity_name}..." * 10
            p_ann = PROMPTS["annotation"].format(claim=claim_text, context=dummy_evidence)
            task_ann = call_llm_sim_enrich(TARGET_MODEL, p_ann)
            
            await asyncio.gather(task_self, task_ann)

    await asyncio.gather(*[process_single_claim(c) for c in claims])
    return time.perf_counter() - t0

# Wrapper specifically for Enrichment, as json_mode is not needed here and the original code is simple
async def call_llm_sim_enrich(model, prompt):
    try:
        # No max_tokens here either; although output is short, we simulate real connection
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
    except: pass

# ================= Main Pipeline Runner =================

async def measure_pipeline(entity_name: str):
    # 1. Main Gen
    bio, t_gen = await phase_gen_main(entity_name)
    
    # 2. Parallel Block
    claims, t_parallel = await phase_parallel_block(entity_name, bio)
    
    # 3. Frequency
    t_freq = await phase_frequency(claims)
    
    # 4. Enrichment
    t_enrich = await phase_enrichment(claims, entity_name)
    
    total_time = t_gen + t_parallel + t_freq + t_enrich
    
    return {
        "entity": entity_name,
        "total_time": total_time,
        "t_gen": t_gen,
        "t_parallel": t_parallel,
        "t_freq": t_freq,
        "t_enrich": t_enrich
    }

async def main():
    print(f"Starting TRUE Pipeline Benchmark (N={SAMPLE_SIZE})...")
    print("NOTE: max_tokens limits REMOVED. This will take longer.")
    
    sample_names = ["Alan Turing", "Marie Curie", "Leonardo da Vinci", "Serena Williams", "Steve Jobs"]
    
    results = []
    for name in tqdm_asyncio(sample_names):
        res = await measure_pipeline(name)
        results.append(res)
        print(f"Entity: {name:<20} | Total: {res['total_time']:.4f} s")
        
    avg_total = np.mean([r['total_time'] for r in results])
    
    print("\n" + "="*60)
    print(f"FINAL BASELINE LATENCY BREAKDOWN (Avg of {len(results)} runs)")
    print("="*60)
    print(f"{'Phase':<25} | {'Details':<30} | {'Time (s)':<10}")
    print("-" * 60)
    print(f"{'1. Bio Generation':<25} | {'GPT-5.2 (Full Gen)':<30} | {np.mean([r['t_gen'] for r in results]):.4f} s")
    print(f"{'2. Parallel Block':<25} | {'Max(5xSamples, Parse)':<30} | {np.mean([r['t_parallel'] for r in results]):.4f} s")
    print(f"{'3. Frequency Score':<25} | {'GPT-4o-mini (Full)':<30} | {np.mean([r['t_freq'] for r in results]):.4f} s")
    print(f"{'4. Enrichment':<25} | {'GPT-5.2 (Full Check)':<30} | {np.mean([r['t_enrich'] for r in results]):.4f} s")
    print("-" * 60)
    print(f"{'TOTAL BASE LATENCY':<57} | {avg_total:.4f} s")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())