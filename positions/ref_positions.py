"""Dump the official (transformers) M-RoPE position ids [T, H, W] for one request, token by token.

usage: PYTHONPATH=pydeps python3 ref_pos.py <model_dir> <image> <16x|4x> <out_tsv> <prompt> [system_prompt]
Output columns: idx, token_id, mm_token_type, T, H, W, token_text
"""
import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL, IMG, MODE, OUT, PROMPT = sys.argv[1:6]
SYS = sys.argv[6] if len(sys.argv) > 6 else None

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()

messages = []
if SYS:
    messages.append({"role": "system", "content": [{"type": "text", "text": SYS}]})
messages.append({"role": "user", "content": [{"type": "image", "image": Image.open(IMG).convert("RGB")},
                                              {"type": "text", "text": PROMPT}]})
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt", downsample_mode=MODE, max_slice_nums=9, enable_thinking=False)
pos, _ = model.model.get_rope_index(inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
                                    target_sizes=inputs.get("target_sizes"), downsample_mode=MODE,
                                    attention_mask=inputs["attention_mask"])
ids = inputs["input_ids"][0].tolist()
mm = inputs["mm_token_type_ids"][0].tolist()
with open(OUT, "w") as f:
    for i, tid in enumerate(ids):
        t, h, w = (int(pos[k, 0, i]) for k in range(3))
        f.write(f"{i}\t{tid}\t{mm[i]}\t{t}\t{h}\t{w}\t{processor.tokenizer.decode([tid])!r}\n")
print("tokens", len(ids), "image tokens", sum(1 for m in mm if m), "->", OUT)
