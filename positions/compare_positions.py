#!/usr/bin/env python3
"""Compare the positions llama.cpp gives each prompt token with the positions from the reference implementation.

inputs:
  reference: TSV from ref_positions.py   (idx, token_id, is_image, T, H, W, token_text)
  llama.cpp: file written by a llama-server built with llama-dump-positions.debug.patch and run with
             LLAMA_DUMP_POS=<file>; one line per token: token_id (-1 for image data), is_image, then the
             position values given to RoPE

The llama.cpp file can start with a few lines from a warm-up batch; they are skipped by aligning on the
first 8 reference tokens. A reference token that llama.cpp does not produce at all (for example a newline
the chat template adds after the image) is reported as "missing in llama.cpp" instead of shifting every
later comparison.

usage: python3 compare_positions.py reference.tsv llama_positions.txt
"""
import sys


def load_ref(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        rows.append({"id": int(f[1]), "img": int(f[2]) != 0, "pos": (int(f[3]), int(f[4]), int(f[5])), "text": f[6]})
    return rows


def load_llama(path):
    rows = []
    for line in open(path):
        f = line.split()
        rows.append({"id": int(f[0]), "img": int(f[1]) != 0, "pos": (int(f[2]), int(f[3]), int(f[4]))})
    return rows


def main():
    ref, lc = load_ref(sys.argv[1]), load_llama(sys.argv[2])
    head = [r["id"] for r in ref[:8]]
    start = next((k for k in range(len(lc)) if [x["id"] for x in lc[k:k + 8]] == head), None)
    if start is None:
        sys.exit("could not find the start of the prompt in the llama.cpp file")
    lc = lc[start:start + len(ref) + 4]
    i = j = 0
    same, diffs, missing = 0, [], []
    while i < len(ref) and j < len(lc):
        r, x = ref[i], lc[j]
        if r["pos"] == x["pos"]:
            same += 1
            i += 1
            j += 1
        elif not r["img"] and r["id"] != x["id"] and x["id"] != -1 and i + 1 < len(ref) and ref[i + 1]["pos"] == x["pos"]:
            missing.append((i, r))
            i += 1
        else:
            diffs.append((i, r, x))
            i += 1
            j += 1
    print(f"reference tokens: {len(ref)}   compared: {same + len(diffs)}   same position: {same}")
    print(f"different position: {len(diffs)}   missing in llama.cpp: {len(missing)}")
    for i, r in missing:
        print(f"  missing  idx {i:5} {r['text']:>14} reference (T,H,W)={r['pos']}")
    for i, r, x in diffs[:40]:
        print(f"  differs  idx {i:5} {r['text']:>14} reference (T,H,W)={r['pos']}  llama.cpp={x['pos']}")
    if len(diffs) > 40:
        print(f"  ... {len(diffs) - 40} more")


if __name__ == "__main__":
    main()
