"""Reference M-RoPE positions for small MiniCPM-V 4.7 canvas layouts, from transformers get_rope_index.

usage: python3 test_expected_positions.py <MiniCPM-V-4.7-1B folder> --no-nl   (needs torch and transformers 5.19)
The model is built on the meta device from its config (no weights needed): get_rope_index only reads the config.
Prints, per layout, the [T, H, W] of every token from the <image> marker to the first text token after the image.
Token layout is the one the processor emits for an image followed by text in the chat template
(render_content joins the parts with "\n"): <image> ov </image> [<slice> s </slice> ... "\n" between rows] "\n" text
"""
import sys

import torch
from transformers import AutoConfig, AutoModelForImageTextToText

# --no-nl: the image is followed directly by the text, without the "\n" of the chat template
WITH_NL = "--no-nl" not in sys.argv[2:]
cfg = AutoConfig.from_pretrained(sys.argv[1])
with torch.device("meta"):
    model = AutoModelForImageTextToText.from_config(cfg)
c = model.config
PAD, IMG, IMG_END, SL, SL_END, NL = c.image_token_id, c.image_start_id, c.image_end_id, c.slice_start_id, c.slice_end_id, c.newline_id
TEXT = 100  # any plain text token
MERGE = 4   # 16x
N_PREFIX = 5


def layout(ov_h, ov_w, n_row, n_col, sl_h, sl_w):
    ids, mm, sizes, names = [TEXT] * N_PREFIX, [0] * N_PREFIX, [], ["text"] * N_PREFIX
    def add(tok, name, n=1, m=0):
        ids.extend([tok] * n); mm.extend([m] * n); names.extend([name] * n)
    add(IMG, "<image>")
    add(PAD, "ov", ov_h * ov_w, 1); sizes.append([ov_h * MERGE, ov_w * MERGE])
    add(IMG_END, "</image>")
    for r in range(n_row):
        if r > 0:
            add(NL, "\\n")
        for _ in range(n_col):
            add(SL, "<slice>")
            add(PAD, "slice", sl_h * sl_w, 1); sizes.append([sl_h * MERGE, sl_w * MERGE])
            add(SL_END, "</slice>")
    if WITH_NL:
        add(NL, "\\n")  # the "\n" the chat template puts between the image and the text
    add(TEXT, "text", 3)
    input_ids = torch.tensor([ids])
    pos, _ = model.model.get_rope_index(input_ids, mm_token_type_ids=torch.tensor([mm]),
                                        target_sizes=torch.tensor(sizes), downsample_mode="16x",
                                        attention_mask=torch.ones_like(input_ids))
    return [(names[i], tuple(int(pos[k, 0, i]) for k in range(3))) for i in range(len(ids))]


for args in ((2, 5, 2, 2, 2, 7), (3, 4, 0, 0, 0, 0)):
    toks = layout(*args)
    i0 = next(i for i, (n, _) in enumerate(toks) if n == "<image>")
    i1 = next(i for i in range(i0 + 1, len(toks)) if toks[i][0] == "text")
    print(f"// layout ov {args[0]}x{args[1]} (h x w), grid {args[2]}x{args[3]} (rows x cols) of {args[4]}x{args[5]} slices")
    print(f"// <image> at index {i0}: {toks[i0][1]}, first text after the image: {toks[i1][1]}")
    print(f"// image chunk ({i1 - i0 - 1} tokens), [T, H, W]:")
    body = toks[i0 + 1:i1]
    for j in range(0, len(body), 8):
        print("    " + " ".join(f"{{{t},{h},{w}}}," for _, (t, h, w) in body[j:j + 8]) + "  // " + " ".join(n for n, _ in body[j:j + 8]))
