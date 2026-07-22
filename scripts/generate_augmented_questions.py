"""
Offline LLM-driven text augmentation for GEAL affordance questions.

Reads the original Affordance-Question.csv (15 questions per (object, affordance) pair),
and generates N additional diverse questions per pair using a local LLM.

Output: Affordance-Question-Augmented.csv with 15+N columns (Question0..Question14+N).

Usage:
    python scripts/generate_augmented_questions.py \
        --csv_path /path/to/Affordance-Question.csv \
        --output_path /path/to/Affordance-Question-Augmented.csv \
        --n_per_pair 50 \
        --llm_model Qwen/Qwen2.5-7B-Instruct

Dependencies:
    pip install transformers pandas torch
"""
import argparse
import os
import re
import time
import warnings
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Prompt Templates
# ============================================================

SYSTEM_PROMPT = """You are a data augmentation assistant for 3D affordance detection.
Your task is to generate diverse, natural language questions that ask about object parts
for specific affordances (functions/actions).

Guidelines:
1. Vary sentence structure — use "Where is", "Which part", "Show me", "Identify", "Point to", "Locate", "Find", "Tell me", etc.
2. Vary vocabulary — use synonyms for the affordance action (e.g., "sit" → "be seated", "take a seat", "rest")
3. Cover different levels of specificity: broad ("the sitting area") vs precise ("the seat cushion")
4. Sound natural, like a human asking about an object
5. Each question must be self-contained and end with a question mark
6. Do NOT include the object name in the question (it will be added by the template later)
7. Do NOT number the questions or use any prefixes

Example for affordance "sit" on object "chair":
- Where is the part designed for sitting?
- Which area of the chair supports a seated person?
- Point to the surface where someone would sit.
- Identify the sitting region on this object.
- Show me the part that bears weight when seated."""

USER_PROMPT_TEMPLATE = """Generate {n} diverse questions for the affordance "{affordance}" on a "{object}" object.
The question should ask about which part of the object enables this function.

Here are some existing examples for reference:
{existing_examples}

Generate {n} NEW questions that are different from the examples above. Each question on a new line:"""


# ============================================================
# Parsing
# ============================================================

def parse_generated_questions(text: str) -> List[str]:
    """Extract question sentences from LLM raw output."""
    questions = []
    for line in text.split("\n"):
        line = line.strip()
        # Remove common prefixes: "1. ", "- ", "* ", '"', "'"
        line = re.sub(r'^[\d\.\-\*\s\"\']+', '', line).strip()
        line = line.strip('"\'')
        if line.endswith("?") and len(line) > 10:
            questions.append(line)
    return questions


def deduplicate_questions(new_qs: List[str], existing_qs: List[str],
                          threshold: float = 0.85) -> List[str]:
    """
    Remove semantically duplicate questions from new_qs by comparing
    against existing_qs using simple token-overlap heuristic.
    (No CLIP dependency needed at generation time.)
    """
    seen = set(q.lower().strip().rstrip("?") for q in existing_qs)
    unique = []
    for q in new_qs:
        key = q.lower().strip().rstrip("?")
        # Token Jaccard overlap with all seen questions
        tokens = set(key.split())
        if not tokens:
            continue
        max_overlap = 0.0
        for s in seen:
            s_tokens = set(s.split())
            union = tokens | s_tokens
            if union:
                overlap = len(tokens & s_tokens) / len(union)
                max_overlap = max(max_overlap, overlap)
        if max_overlap < threshold:
            seen.add(key)
            unique.append(q)
    return unique


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate_questions(llm, tokenizer, object_name: str, affordance: str,
                       existing: List[str], n: int = 50,
                       max_new_tokens: int = 128, temperature: float = 0.9,
                       device: str = "cuda:0") -> List[str]:
    """Generate N diverse questions for one (object, affordance) pair."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            n=n, affordance=affordance, object=object_name,
            existing_examples="\n".join(f"- {q}" for q in existing[:5]),
        )},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return parse_generated_questions(raw)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LLM-driven text augmentation for affordance questions")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to original Affordance-Question.csv")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save augmented CSV")
    parser.add_argument("--n_per_pair", type=int, default=50,
                        help="Number of new questions per (object, affordance) pair")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID for generation")
    parser.add_argument("--max_attempts", type=int, default=5,
                        help="Max LLM calls per pair to reach n_per_pair")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of pairs to generate per LLM call (currently 1)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for LLM inference")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Model dtype: float16 or 8bit")
    args = parser.parse_args()

    # ---- Load CSV ----
    df = pd.read_csv(args.csv_path)
    existing_cols = [c for c in df.columns if c.startswith("Question")]
    print(f"[INFO] 加载 {len(df)} 行, {len(existing_cols)} 个已有问题列")
    print(f"       列名: {list(df.columns)}")

    # ---- Load LLM ----
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.dtype == "8bit":
        print("[INFO] 使用 8-bit 量化加载 LLM...")
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

    # ---- Prepare new columns ----
    new_col_start = len(existing_cols)
    new_col_names = [f"Question{i}" for i in range(new_col_start, new_col_start + args.n_per_pair)]
    for col in new_col_names:
        df[col] = ""

    # ---- Generate per row ----
    n_ok = 0
    n_skip = 0
    t_start = time.time()

    for idx, row in df.iterrows():
        obj = row["Object"]
        aff = row["Affordance"]
        existing = [row[c] for c in existing_cols if pd.notna(row[c])]

        if not existing:
            print(f"  [{idx+1}/{len(df)}] ({obj}, {aff}) ⚠️ 无已有问题,跳过")
            n_skip += 1
            continue

        # Collect new questions, possibly over multiple LLM calls
        all_new = []
        seen_set = set(existing)
        for attempt in range(args.max_attempts):
            remaining = args.n_per_pair - len(all_new)
            if remaining <= 0:
                break
            # Ask for more than needed to account for duplicates
            raw_qs = generate_questions(
                llm, tokenizer, obj, aff, existing,
                n=remaining * 2,  # ask for 2x to have buffer
                temperature=0.9 + 0.05 * attempt,  # increase diversity on retry
                device=args.device,
            )
            # Deduplicate against existing
            for q in raw_qs:
                if q not in seen_set:
                    seen_set.add(q)
                    all_new.append(q)
            print(f"    Attempt {attempt+1}: 生成 {len(raw_qs)} 条, "
                  f"去重后累计 {len(all_new)}/{args.n_per_pair}")

        # Fill DataFrame
        for i, q in enumerate(all_new[:args.n_per_pair]):
            df.at[idx, new_col_names[i]] = q

        if len(all_new) >= args.n_per_pair:
            n_ok += 1
        else:
            n_skip += 1
            print(f"    ⚠️ 仅生成 {len(all_new)}/{args.n_per_pair}, 不足部分留空")

        # Periodic save
        if (idx + 1) % 10 == 0:
            df.to_csv(args.output_path, index=False)
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed * 60
            print(f"  [SAVE] 已处理 {idx+1}/{len(df)} 行, "
                  f"速率 {rate:.1f} 行/分钟, 已保存到 {args.output_path}")

    # Final save
    df.to_csv(args.output_path, index=False)
    elapsed = time.time() - t_start
    print(f"\n[DONE] 完成! 输出: {args.output_path}")
    print(f"       总行数: {len(df)}, 问题列: {len([c for c in df.columns if c.startswith('Question')])}")
    print(f"       成功: {n_ok}, 不足: {n_skip}")
    print(f"       总耗时: {elapsed/60:.1f} 分钟")


if __name__ == "__main__":
    main()