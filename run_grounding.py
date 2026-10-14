#!/usr/bin/env python3
"""MiniCPM-V 4.7 GUI grounding check against a llama-server (OpenAI-compatible) endpoint.

Sends 18 grounding questions (2 synthetic screenshots, known target boxes), then reports:
  - hits: predicted box/point centre inside the target box
  - mean dy: mean vertical offset of predicted centres from target centres, in 0-1000 units
    (positive = predicted too low)
  - with --reference: median distance (0-1000 units) to answers from another deployment

Standard library only.

usage:
  python3 run_grounding.py --server http://127.0.0.1:8080 --out answers.jsonl
  python3 run_grounding.py --server http://127.0.0.1:8080 --out answers.jsonl --reference reference/official-vllm-4x.jsonl
  python3 run_grounding.py --score-only answers.jsonl --reference reference/official-vllm-4x.jsonl
"""
import argparse
import base64
import json
import re
import statistics
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM_PROMPT = ("You are MiniCPM-V 4.7, a multimodal large language model developed by ModelBest "
                 "and the OpenBMB community.")  # system prompt used by the official deployment


def coords(text, n):
    """First bracketed group of n numbers, else the last 4 (bbox) / first 2 (point) numbers."""
    pat = r"[\[(]\s*" + r"\s*,\s*".join([r"(-?\d+(?:\.\d+)?)"] * n) + r"\s*[\])]"
    m = re.search(pat, text)
    if m:
        return [float(v) for v in m.groups()]
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(nums) < n:
        return None
    return nums[-4:] if n == 4 else nums[:2]


def centre(case, answer):
    """Predicted centre in 0-1000 units. Point prompts sometimes get a 4-number box back."""
    box = coords(answer, 4)
    if box and (case["answer_format"] == "bbox" or answer.count(",") >= 3):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    pt = coords(answer, 2)
    return (pt[0], pt[1]) if pt else None


def target(case):
    x1, y1, x2, y2 = case["gt_box_px"]
    w, h = case["width"], case["height"]
    return x1 * 1000 / w, y1 * 1000 / h, x2 * 1000 / w, y2 * 1000 / h


def ask(server, case):
    b64 = base64.b64encode((HERE / case["image"]).read_bytes()).decode()
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": case["prompt"]},
            ]},
        ],
        "temperature": 0,
        "max_tokens": 256,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"].get("content") or ""


def load_answers(path):
    return {r["id"]: r["answer"] for r in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())}


def report(cases, answers, reference=None):
    hits, dys, dists = 0, [], []
    print(f"{'id':4} {'hit':3} {'centre':>14} {'target centre':>14}  answer")
    for c in cases:
        a = answers.get(c["id"], "")
        p = centre(c, a)
        tx1, ty1, tx2, ty2 = target(c)
        tc = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
        hit = bool(p) and tx1 <= p[0] <= tx2 and ty1 <= p[1] <= ty2
        hits += hit
        if p:
            dys.append(p[1] - tc[1])
        if reference and p:
            q = centre(c, reference.get(c["id"], ""))
            if q:
                dists.append(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5)
        pc = f"({p[0]:.0f}, {p[1]:.0f})" if p else "unparsed"
        print(f"{c['id']:4} {'yes' if hit else 'no':3} {pc:>14} ({tc[0]:.0f}, {tc[1]:.0f})  {a.strip()[:60]!r}")
    print()
    print(f"hits: {hits}/{len(cases)}")
    print(f"mean dy (0-1000, + = too low): {statistics.mean(dys):+.1f}" if dys else "mean dy: n/a")
    if reference:
        print(f"median distance to reference (0-1000): {statistics.median(dists):.1f}  (max {max(dists):.1f}, n={len(dists)})"
              if dists else "median distance to reference: n/a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", help="llama-server base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--out", help="write answers here (jsonl)")
    ap.add_argument("--score-only", help="score an existing answers file instead of querying a server")
    ap.add_argument("--reference", help="answers from another deployment to compare against (jsonl)")
    ap.add_argument("--cases", default=str(HERE / "cases.json"))
    ap.add_argument("--only", help="comma-separated case ids to run, e.g. G1")
    args = ap.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    reference = load_answers(args.reference) if args.reference else None
    if args.score_only:
        report(cases, load_answers(args.score_only), reference)
        return
    if not args.server:
        sys.exit("--server or --score-only is required")
    answers = {}
    out = open(args.out, "w") if args.out else None
    for c in cases:
        answers[c["id"]] = ask(args.server, c)
        if out:
            out.write(json.dumps({"id": c["id"], "answer": answers[c["id"]]}, ensure_ascii=False) + "\n")
            out.flush()
    report(cases, answers, reference)


if __name__ == "__main__":
    main()
