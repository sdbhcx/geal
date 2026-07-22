"""
Offline LLM-driven functional description generation for GEAL affordance pairs.

For each (object, affordance) pair, generates a 1-2 sentence functional definition
describing what the affordance means in physical interaction terms.

Output: Affordance-Functional-Desc.csv with columns: Object, Affordance, FunctionalDesc.

Usage:
    python scripts/generate_functional_descriptions.py \
        --csv_path /path/to/Affordance-Question.csv \
        --output_path /path/to/Affordance-Functional-Desc.csv \
        --llm_model Qwen/Qwen2.5-7B-Instruct

Dependencies:
    pip install transformers pandas torch
"""
import argparse
import os
import time
import warnings
from typing import List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Prompt Templates
# ============================================================

SYSTEM_PROMPT = """You are an expert in functional object understanding for 3D affordance detection.
For each (object, affordance) pair, write a concise functional definition (1-2 short sentences, under 20 words each) that describes:

1. What the affordance means as a physical human action
2. What physical properties of the object part enable this affordance
3. How the human interacts with the part

The definition will be used as prefix context for a CLIP text encoder to better distinguish between different affordances. Be precise, concrete, and avoid generic statements.

Output format: Just the definition text, nothing else. No prefixes, labels, or numbering."""

USER_PROMPT_TEMPLATE = """Write a functional definition for the affordance "{affordance}" on a "{object}" object.

Example for (chair, sit):
"Sitting involves resting one's body weight on a flat, horizontal surface that is elevated from the ground. The seat must be broad enough to support the buttocks and thighs."

Now write for ({object}, {affordance}):"""


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate_description(llm, tokenizer, object_name: str, affordance: str,
                         device: str = "cuda:0",
                         max_new_tokens: int = 80) -> str:
    """Generate one functional description for a single (object, affordance) pair."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            affordance=affordance, object=object_name,
        )},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,  # low temperature for factual consistency
        top_p=0.95,
        do_sample=False,  # deterministic for quality
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    desc = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Clean up: remove quotes, extra whitespace
    desc = desc.strip('"\' \n')
    return desc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate functional descriptions for affordance pairs"
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to Affordance-Question.csv (used for (obj, aff) pairs)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save Affordance-Functional-Desc.csv")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID for generation")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for LLM inference")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Model dtype: float16 or 8bit")
    args = parser.parse_args()

    # ---- Load CSV ----
    src_df = pd.read_csv(args.csv_path)
    # Extract unique (Object, Affordance) pairs
    pairs = src_df[["Object", "Affordance"]].drop_duplicates().reset_index(drop=True)
    print(f"[INFO] 加载 {len(src_df)} 行, 去重后有 {len(pairs)} 个 (Object, Affordance) 对")

    # ---- Load LLM ----
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.dtype == "8bit":
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_model, load_in_8bit=True, device_map=args.device,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_model, torch_dtype=dtype, device_map=args.device,
        )
    llm.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[INFO] LLM 加载完成: {args.llm_model} ({args.dtype})")

    # ---- Generate ----
    results = []
    t_start = time.time()

    for idx, row in pairs.iterrows():
        obj = row["Object"]
        aff = row["Affordance"]

        desc = generate_description(
            llm, tokenizer, obj, aff, device=args.device,
        )
        results.append({"Object": obj, "Affordance": aff, "FunctionalDesc": desc})

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed * 60
            print(f"  [{idx+1}/{len(pairs)}] ({obj}, {aff}) → {desc[:60]}... "
                  f"({rate:.1f} 对/分钟)")

    # ---- Save ----
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_path, index=False)
    elapsed = time.time() - t_start
    print(f"\n[DONE] 完成! 输出: {args.output_path}")
    print(f"       总行数: {len(out_df)}")
    print(f"       总耗时: {elapsed/60:.1f} 分钟")
    print(f"\n前 5 条预览:")
    print(out_df.head().to_string())


if __name__ == "__main__":
    main()