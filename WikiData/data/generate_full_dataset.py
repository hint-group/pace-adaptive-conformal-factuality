import json
import pickle
import re
import asyncio
import os
import wikipedia
import numpy as np
import logging
from typing import List, Dict, Any
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from rank_bm25 import BM25Okapi

# ================= 1. Global Configuration =================

API_KEY = ""
BASE_URL = "https://api.openai.com/v1"

TARGET_MODEL = "gpt-5.2"       
HELPER_MODEL = "gpt-4o-mini"   

NUM_SAMPLES = 5                
MAX_CONCURRENT_ENTITIES = 2    
MAX_CONCURRENT_CLAIMS = 5      

INPUT_FILE = "../ProcessedData/factscore_final_dataset.pkl"
OUTPUT_FILE = "./factscore_dataset_v3_p3.jsonl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# [New] Global circuit breaker: LogProbs enabled by default, globally disabled upon error
ENABLE_LOGPROBS = True 

# ================= 2. Prompts =================

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

    # LogProb Prompt (Only used when ENABLE_LOGPROBS=True)
    "log_prob": """You will get a claim and the original prompt that motivated it. 
Is this claim True or False? 
Answer with a SINGLE character: T or F. Do not output anything else.
Original prompt: {prompt}
Claim: {claim}
Answer:""",

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

# ================= 3. Utility Functions (With error handling) =================

async def call_llm(model: str, prompt: str, temp: float = 0.0, json_mode: bool = False, logprobs: bool = False):
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "timeout": 40.0
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        # Add parameters only if global switch is on and function requests logprobs
        if logprobs and ENABLE_LOGPROBS:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
            kwargs["max_tokens"] = 1 
        
        return await client.chat.completions.create(**kwargs)
        
    except Exception as e:
        error_msg = str(e)
        # [Critical] Identify errors where logprobs are unsupported
        if "400" in error_msg and ("Unsupport" in error_msg or "parameter" in error_msg or "logprobs" in error_msg):
            raise ValueError("LogprobsUnsupported") # Raise specific exception for upstream handling
        
        logger.error(f"API Error ({model}): {error_msg[:100]}")
        return None

# Wiki Section
async def get_wiki_context(entity_name: str) -> str:
    try:
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _wiki_worker(entity_name)), timeout=10.0
        )
    except: return ""

def prepare_bm25(full_text: str):
    """
    [New] Preprocess Wiki text: Chunk paragraphs and build BM25 index
    Returns: (bm25_object, chunk_list)
    """
    if not full_text:
        return None, []
    
    # 1. Chunking
    # Split by double newlines, representing natural paragraphs in Wiki
    # Filter out short noisy paragraphs (<50 chars)
    chunks = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]
    
    if not chunks:
        return None, []
        
    # 2. Tokenization
    # Simple space tokenization is sufficient for English
    tokenized_corpus = [doc.split(" ") for doc in chunks]
    
    # 3. Build Index
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks

def query_bm25(bm25, chunks, query: str, top_k: int = 3) -> str:
    """
    [New] Retrieve most relevant paragraphs using BM25
    """
    if not bm25 or not chunks:
        return ""
        
    tokenized_query = query.split(" ")
    top_chunks = bm25.get_top_n(tokenized_query, chunks, n=top_k)
    return "\n...\n".join(top_chunks)

def _wiki_worker(name):
    try:
        r = wikipedia.search(name)
        if not r: return ""
        return wikipedia.page(r[0], auto_suggest=False).content
    except: return ""

# ================= 4. Feature Calculation (With circuit breaker) =================

async def calc_self_eval(prompt: str, claim: str) -> float:
    # Force JSON mode parsing
    res = await call_llm(TARGET_MODEL, PROMPTS["self_eval"].format(prompt=prompt, claim=claim), temp=0.0) # Removed json_mode=True as prompt returns jsonl. Modify prompt if strict json object is needed.
    if not res: return 0.5
    try:
        content = res.choices[0].message.content
        match = re.search(r"gpt-score\":\s*([\d\.]+)", content)
        return float(match.group(1)) if match else 0.5
    except: return 0.5

async def calc_log_prob(prompt: str, claim: str) -> float:
    """
    Calculate Token Probabilities. Includes global circuit breaker.
    """
    global ENABLE_LOGPROBS # Reference global variable

    # [1. Check Circuit Breaker] If global switch is off, skip request
    if not ENABLE_LOGPROBS:
        return 0.0

    try:
        # [2. Attempt Call]
        res = await call_llm(TARGET_MODEL, PROMPTS["log_prob"].format(prompt=prompt, claim=claim), temp=0.0, logprobs=True)
        
        if not res: return 0.0
        
        # 提取 Logprobs
        top_logprobs = res.choices[0].logprobs.content[0].top_logprobs
        token_map = {tl.token.strip(): tl.logprob for tl in top_logprobs}
        
        true_logprob = -999.0
        found = False
        for key, val in token_map.items():
            if key.lower() in ['t', 'true']:
                true_logprob = max(true_logprob, val)
                found = True
        
        if found and true_logprob > -100:
            return np.exp(true_logprob)
        return 0.0

    except ValueError as e:
        # [3. Trigger Circuit Breaker] Catch specific error, permanently disable switch
        if str(e) == "LogprobsUnsupported":
            if ENABLE_LOGPROBS: # Prevent duplicate logging
                logger.warning("⚠️ Model does not support logprobs. Globally disabling log_prob calculation for future samples.")
                ENABLE_LOGPROBS = False
            return 0.0
        return 0.0
    except Exception:
        return 0.0

async def calc_annotation(claim: str, context: str) -> int:
    if not context: return 0
    res = await call_llm(TARGET_MODEL, PROMPTS["annotation"].format(claim=claim, context=context), temp=0.0)
    if not res: return 0
    txt = res.choices[0].message.content.strip()
    return 1 if "1" in txt and "-1" not in txt else 0

# ================= 5. Pipeline =================

async def process_entity(prompt: str, name: str, entity_sem: asyncio.Semaphore) -> Dict:
    async with entity_sem:
        logger.info(f"[{name}] Generating bio...")
        res_base = await call_llm(TARGET_MODEL, prompt, temp=1.0)
        if not res_base: return None
        response_text = res_base.choices[0].message.content

        # 1. Concurrent Samples, Parsing, Wiki
        logger.info(f"[{name}] Samples/Wiki/Parsing...")
        t_samples = [call_llm(TARGET_MODEL, prompt, temp=1.0) for _ in range(NUM_SAMPLES)]
        # Parsing uses Helper Model
        t_parse = call_llm(HELPER_MODEL, PROMPTS["separator"].format(input_text=response_text), temp=0.0, json_mode=True)
        t_wiki = get_wiki_context(name)
        
        results_samples = await asyncio.gather(*t_samples)
        res_parse, wiki_full_text = await asyncio.gather(t_parse, t_wiki)
        
        # [New] --- Build BM25 Index (Once) ---
        bm25_obj, wiki_chunks = prepare_bm25(wiki_full_text)

        samples_texts = [r.choices[0].message.content for r in results_samples if r]

        # 2. Parse Claims
        claims_list = []
        if res_parse:
            try:
                raw_claims = json.loads(res_parse.choices[0].message.content).get("claims", [])
                claims_list = [{"id": i, "message": str(c)} for i, c in enumerate(raw_claims)]
            except: pass
        
        if not claims_list: return None
        logger.info(f"[{name}] {len(claims_list)} claims. Calculating scores...")

        # 3. Frequency Score (Helper Model)
        claim_str = "\n".join([f"ID {c['id']}: {c['message']}" for c in claims_list])
        freq_tasks = []
        for s_text in samples_texts:
            f_prompt = PROMPTS["frequency"].format(claim_string=claim_str, sample_text=s_text)
            freq_tasks.append(call_llm(HELPER_MODEL, f_prompt, temp=0.0, json_mode=True))
        
        res_freqs = await asyncio.gather(*freq_tasks)
        
        support_counts = {c['id']: 0 for c in claims_list}
        for rf in res_freqs:
            if not rf: continue
            try:
                scores = json.loads(rf.choices[0].message.content).get("scores", [])
                for item in scores:
                    if item.get("score") == 1:
                        support_counts[item.get("id")] += 1
            except: pass

        # 4. Feature Calculation (Target Model)
        claim_sem = asyncio.Semaphore(MAX_CONCURRENT_CLAIMS) 
        
        async def _enrich(c):
            async with claim_sem: 
                # [New] --- Dynamic Evidence Retrieval ---
                # Retrieve Wiki using current Claim text
                if bm25_obj:
                    # Get Top 3 relevant paragraphs as Evidence
                    retrieved_evidence = query_bm25(bm25_obj, wiki_chunks, c['message'], top_k=3)
                else:
                    retrieved_evidence = ""
                # ------------------------------

                # Checks ENABLE_LOGPROBS
                t_self = calc_self_eval(prompt, c['message'])
                t_log = calc_log_prob(prompt, c['message']) 
                
                # [Modified] Pass retrieved Evidence instead of full wiki_context
                t_label = calc_annotation(c['message'], retrieved_evidence)
                
                s_eval, s_log, label = await asyncio.gather(t_self, t_log, t_label)
                freq = support_counts.get(c['id'], 0) / max(len(samples_texts), 1)
                
                scores = {"self_eval": s_eval, "frequency": freq}
                if ENABLE_LOGPROBS:
                    scores["log_prob"] = s_log

                return {
                    "id": c['id'],
                    "message": c['message'],
                    "scores": scores,
                    "label": label,
                    "evidence_used": retrieved_evidence[:500] + "..." # [Optional] Log used evidence for debugging
                }

        enriched_claims = await asyncio.gather(*[_enrich(c) for c in claims_list])
        
        logger.info(f"[{name}] Done.")
        return {
            "entity_name": name,
            "prompt": prompt,
            "response": response_text,
            # [Modified] Save first 2000 chars as preview; actual judgment uses dynamic retrieval
            "wiki_context_excerpt": wiki_full_text[:2000] + "..." if wiki_full_text else "",
            "claims": list(enriched_claims)
        }

# ================= Main =================
async def main():
    if not os.path.exists(INPUT_FILE): return
    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)[2875:6000] 

    sem = asyncio.Semaphore(MAX_CONCURRENT_ENTITIES)
    tasks = []
    for entry in data:
        # Handle entry as text or dict
        full_prompt = entry['prompt'] if isinstance(entry, dict) else entry
        match = re.search(r"about (.*?)\.", full_prompt)
        name = match.group(1) if match else "Unknown"
        tasks.append(process_entity(full_prompt, name, sem))
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for fut in tqdm_asyncio.as_completed(tasks):
            res = await fut
            if res:
                f.write(json.dumps(res) + "\n")
                f.flush()

if __name__ == "__main__":
    asyncio.run(main())