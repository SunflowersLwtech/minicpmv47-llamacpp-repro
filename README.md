# MiniCPM-V 4.7 in llama.cpp: image marker positions differ from the official code

This folder reproduces a difference between llama.cpp and the official MiniCPM-V 4.7 code, and checks a fix.
MiniCPM-V 4.7 is an image and text model from OpenBMB. llama.cpp supports it since PR #29416 (commit `69f201a`).

## What goes wrong

Ask the model where something is on a screenshot, for example "give the box of the Save button".
With llama.cpp the answer is often placed too low.
The official demo of the same model (Hugging Face Space `openbmb/MiniCPM-V-4.7-Demo`) does not show this.

## Why

Before the model reads an image, the image is turned into a small overview plus a grid of tiles.
In the prompt, these pieces are separated by marker tokens: `</image>`, `<slice>`, `</slice>`,
and a newline at the end of each row of tiles.

Every token in the prompt also gets a position: three numbers that tell the model when and where the token is
(a time value, a row, and a column).

- Official code: each marker gets the row and column of the tile corner it belongs to.
- llama.cpp: the markers are numbered one after another, like normal text.
  To the model they then sit on a diagonal line across the image.

The positions of the image pieces themselves are the same in both.
This is why most image questions still work, and why questions about where things are do not.

## Result

18 questions on 2 synthetic screenshots: a settings window and a toolbar with 24 similar buttons.
A question counts as a hit when the centre of the answer falls inside the target box.
Coordinates use the model's own 0-1000 scale.

Model: MiniCPM-V-4.7-35B-A3B, Q4_K_M weights. Image file: `mmproj-model-f16-4x.gguf`.
This image file keeps 4 times more image detail than the default one.

| | hits | mean vertical offset | distance to official answers (median) |
|---|---|---|---|
| llama.cpp master before the fix (b11609) | 3/18 | +60.3 (too low) | 79.8 |
| llama.cpp with the fix | 15/18 | -1.2 | 10.0 |
| official deployment, for reference | 14/18 | -0.9 | 0 |

The official answers were collected on 2026-10-14 from the official demo (Hugging Face Space
`openbmb/MiniCPM-V-4.7-Demo`), with greedy decoding (temperature 0).
They are in `reference/official-vllm-4x.jsonl`.

With the default image file (`mmproj-model-f16.gguf`), the model locates things poorly in all setups:
2/18 before the fix, 3/18 after, and 3/18 for the official deployment.
That part is the model, not this problem.

## Reproduce

### 1. Unit test, no model needed (a few minutes on CPU)

`patches/minicpmv47-canvas-pos-test-only.patch` adds the test and leaves the position code unchanged.
`patches/minicpmv47-canvas-pos-fix-on-test-only.patch` then changes the position code.
Together they equal `patches/minicpmv47-canvas-pos.patch`, the full change.
The expected values in the test were computed with the official code (`get_rope_index` in transformers 5.19).

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git apply <this folder>/patches/minicpmv47-canvas-pos-test-only.patch
cmake -B build && cmake --build build --target test-mtmd-impl -j 8
./build/bin/test-mtmd-impl test_decoder_pos_minicpmv4_7   # fails: 13 of 182 checks
git apply <this folder>/patches/minicpmv47-canvas-pos-fix-on-test-only.patch
cmake --build build --target test-mtmd-impl -j 8
./build/bin/test-mtmd-impl test_decoder_pos_minicpmv4_7   # passes: 182 of 182
```

Full output of both runs (on master `318a471`): `expected/unit-test-master-plus-test.txt` and `expected/unit-test-with-fix.txt`.

The expected positions written in the test come from `positions/test_expected_positions.py`.
It builds the model from its config only (no weights are loaded) and prints the positions that `get_rope_index`
gives to the two small layouts used in the test. Its output is in `positions/test_expected_positions.out`:

```sh
python3 positions/test_expected_positions.py <MiniCPM-V-4.7-1B folder> --no-nl
```

### 2. End to end (needs the 21 GB model and about 25 GB of GPU memory)

Model files: `MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf` and `mmproj-model-f16-4x.gguf`,
both from https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf.

```sh
llama-server -m MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf --mmproj mmproj-model-f16-4x.gguf \
  --jinja -c 16384 -ngl 999 -fa on --port 8080
python3 run_grounding.py --server http://127.0.0.1:8080 --out answers.jsonl \
  --reference reference/official-vllm-4x.jsonl
```

The script sends the same system prompt as the official deployment.
It uses temperature 0 and turns thinking off.
Only the Python standard library is needed.
`expected/` holds the full output of our two runs (before and after the fix).

### 3. Position check, token by token (needs the 1B model and transformers 5.19)

1. Apply `positions/llama-dump-positions.debug.patch` (for checking only).
   With it, llama-server writes the position of every prompt token to a file.
2. Start the server with the 1B model and the dump turned on:

   ```sh
   LLAMA_DUMP_POS=llama_positions.txt llama-server -m MiniCPM-V-4_7-1B-F16.gguf \
     --mmproj mmproj-model-f16.gguf --jinja -ngl 0 --port 8080
   ```

3. Send one question, then get the official positions for the same prompt and compare:

   ```sh
   python3 run_grounding.py --server http://127.0.0.1:8080 --only G1
   python3 positions/ref_positions.py <MiniCPM-V-4.7-1B folder> images/gui_settings.png 16x reference.tsv \
     "<the G1 prompt from cases.json>" "<the system prompt from run_grounding.py>"
   python3 positions/compare_positions.py reference.tsv llama_positions.txt
   ```

Our result on the 528-token prompt:

- Before the fix, 44 tokens are at different positions:
  the `<image>` marker, the 14 markers between image pieces, and the 29 text tokens after the image.
- With the fix, 1 token is at a different position (the `<image>` marker), and 1 token is missing (the newline after the image).
  Both are described below.

## What the fix changes

Two functions in `tools/mtmd/mtmd.cpp`:

- `mtmd_canvas_decoder_pos`: markers get the row and column of their tile corner, as in the official code.
  Their time value is fixed to the start of the image.
- `mtmd_image_tokens_get_n_pos`: the text after the image starts at the same position as in the official code.

The test needs to build the image layout without loading a model.
For this, the code that builds that layout is moved, unchanged, into `mtmd_canvas_image_tokens()`.
A small helper for tests is added in `mtmd-internal.h`.

Only the marker change makes a measurable difference.
With only the text start changed: 3/18. With only the marker change: 15/18.

## Known differences that remain

- **Newline after the image.**
  The official chat template puts a newline between an image and the text that follows it.
  llama.cpp builds the prompt without it, so llama.cpp has one token less in that case.
  All other tokens keep the official positions.
  A version that adds this newline was also tested: it gives the same results, but it adds a newline where the
  official code has none (an image with no text after it). It is not included.
- **The `<image>` marker** is one row and one column further than in the official code.
  It sits in the text before the image and is not changed here.
- **Large tile grids in 16x mode.**
  When the grid of tiles is large (for example 3 x 3 tiles of 8 x 8), the markers need more positions than the
  image area provides. The text after the image then starts a few positions later than in the official code,
  and a warning is printed. None of our test images reached this.
- **More than one image in one message** was not tested.

## Test setup

- AMD Ryzen AI Max+ 395 (Radeon 8060S), Vulkan backend, Ubuntu 24.04.
- llama.cpp b11609 (`fcac190`), and the fix on master `8c2134a`.
  The files the fix touches did not change up to `318a471`.
- Official code: transformers 5.19 (CPU, fp32) for positions; the official demo for answers.

## Files

| file | content |
|---|---|
| `cases.json` | the 18 questions, images and target boxes (pixels) |
| `images/` | the 2 screenshots |
| `run_grounding.py` | sends the questions and scores the answers |
| `reference/` | answers from the official deployment |
| `expected/` | our answers, reports and unit test output, before and after the fix |
| `patches/` | the full change, and the same change split into test only + fix |
| `positions/` | debug patch and scripts for the token-by-token check, and the script that makes the test's expected positions |

Every result above comes from running the steps on this page.
