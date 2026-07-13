import json
import os
import asyncio
from typing import List, Dict
from pyserini.search.lucene import LuceneSearcher
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ================= Configuration =================
# 1. Wikipedia Index Path (Replace with your actual path)
# If no local index, Pyserini supports downloading prebuilt indexes like 'wikipedia-dpr-100w'.
# For FActScore reproduction, it is best to use the full enwiki.
INDEX_PATH = "path/to/your/wikipedia-index" 

# 2. OpenAI Configuration
API_KEY = ""
BASE_URL = "https://api.openai.com/v1"
JUDGE_MODEL = "gpt-5.2"  # Judge Model

INPUT_FILE = "factscore_gpt-5.2.json"
OUTPUT_FILE = "factscore_gpt-5.2_annotated.json"

# Initialize Searcher (BM25)
# If no local index, use Pyserini's prebuilt index for testing:
# searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
try:
    searcher = LuceneSearcher(INDEX_PATH)
except Exception as e:
    print(f"Index load failed: {e}")
    print("For testing, using a prebuilt index (WARNING: Results may not be optimal for FactScore)")
    searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= Core Logic 1: BM25 Retrieval =================

def retrieve_evidence(query: str, k: int = 5) -> str:
    """
    Retrieve Top-K relevant passages using BM25
    """
    try:
        hits = searcher.search(query, k=k)
        passages = []
        for hit in hits:
            # hit.raw is the original document content (JSON string)
            # Adjust parsing logic based on index structure
            # Typically 'contents' field in Pyserini index
            try:
                # Explicitly get Document object, then raw
                raw_doc = searcher.doc(hit.docid).raw()
                doc = json.loads(raw_doc)
                text = doc.get('contents', '') or doc.get('text', '')
            except:
                # Fallback
                try:
                    text = searcher.doc(hit.docid).raw()
                except:
                    text = ""
            
            passages.append(f"- {text.strip()}")
        
        return "\n".join(passages)
    except Exception as e:
        print(f"Retrieval error for query '{query}': {e}")
        return ""

# ================= Core Logic 2: LLM Judge =================

def get_judge_prompt(claim: str, evidence: str) -> str:
    # Reference: Cherian (2024) and FActScore Prompt Design
    return f"""You are an expert fact-checker. You will be given a claim and a set of retrieved Wikipedia passages as evidence.
Your task is to determine if the claim is supported by the evidence.

Evidence:
{evidence}

Claim:
{claim}

Is the claim supported by the evidence? 
Respond with ONLY a JSON object: {{"label": "S"}} if Supported, or {{"label": "NS"}} if Not Supported (contradicted or unrelated).
Do not output anything else."""

async def verify_claim(claim_text: str, entity_name: str) -> str:
    """
    Verify claim against entity name. Returns "S" (Supported) or "NS" (Not Supported)
    """
    # 1. Construct Retrieval Query
    # Standard FActScore strategy: Entity Name + Claim Keywords
    # To improve recall: "Entity Name claim_text"
    query = f"{entity_name} {claim_text}"
    
    # 2. Execute Retrieval
    evidence_text = retrieve_evidence(query, k=5)
    
    if not evidence_text:
        return "NS" # No evidence found, default to Not Supported

    # 3. LLM Judge
    prompt = get_judge_prompt(claim_text, evidence_text)
    
    try:
        response = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"} # Enforce JSON format
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get("label", "NS")
    except Exception as e:
        print(f"Judge error: {e}")
        return "NS"

# ================= Main Process =================

async def process_entry(entry: Dict):
    # Extract Entity Name from prompt
    # Prompt format: "Please write one biographical paragraph about Suthida."
    # Simple extraction: remove prefix and suffix
    prompt_text = entry.get("prompt", "")
    entity_name = prompt_text.replace("Please write one biographical paragraph about ", "").replace(".", "").strip()
    
    claims = entry.get("claims", [])
    if not claims:
        return entry

    # Verify each claim
    # Verify claims serially for a single entry, but entries are processed in parallel
    for claim in claims:
        # Fill in "annotation"
        label = await verify_claim(claim["message"], entity_name)
        
        # Map FActScore standard S/NS to requirements
        claim["annotation"] = label 

    return entry

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    print("Loading data...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Concurrency control
    semaphore = asyncio.Semaphore(10) # Adjust based on API Rate Limit

    async def sem_process(e):
        async with semaphore:
            return await process_entry(e)

    print(f"Auto-annotating {len(data)} entries using BM25 + {JUDGE_MODEL}...")
    tasks = [sem_process(entry) for entry in data]
    new_data = await tqdm_asyncio.gather(*tasks)

    # Save
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Annotated data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())