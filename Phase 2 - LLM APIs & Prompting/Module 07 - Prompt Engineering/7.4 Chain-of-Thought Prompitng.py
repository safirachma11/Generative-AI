import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# Without CoT - model jumps to answer, more likely to be wrong
DIRECT_PROMPT = "If a model costs $3.00 per million input tokens and $15.00 per million output tokens, and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?"

# With CoT - model reasons through each step
COT_PROMPT = """If a model costs $3.00 per million input tokens and $15.00 per million output tokens,
and a request uses 2,400 input tokens and 800 output tokens,
what is the total cost in USD?
Think through this step by step before giving the final answer."""

# Zero-shot CoT: just adding "think step by step"
ZERO_SHOT_COT = """Solve this problem. Think step by step, showing each calculation.
Finally, state: ANSWER: $X.XXXXXX
Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens
and 400 output tokens. The model costs $3.00/M input and $15.00/M output.
What is the daily cost?"""

import time

for label, prompt in [
    ("Direct", DIRECT_PROMPT),
    ("CoT", COT_PROMPT),
    ("Zero-shot CoT", ZERO_SHOT_COT)
]:
    try:
        resp = client.chat.completions.create(
            model="openrouter/free", # Menggunakan auto-router untuk model gratis
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        print(f"==={label} ===")
        if resp and resp.choices:
            content = resp.choices[0].message.content[:300]
            # Hindari error charmap di Windows (cp1252)
            safe_content = content.encode('ascii', errors='replace').decode('ascii')
            print(safe_content)
        else:
            print(f"Unexpected response format: {resp}")
        print()
    except Exception as e:
        print(f"==={label} ===")
        print(f"ERROR: {e}\n")
    
    # Beri jeda 3 detik agar tidak terkena rate limit
    time.sleep(3)