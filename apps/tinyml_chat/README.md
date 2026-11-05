# TinyML Chat

A from-scratch LLM chat application that runs on a personal **laptop CPU** (no Colab, no GPU,
nothing to compile), demonstrating the optimization ideas from the exercises applied to a real model.

Follows the idea of [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine) /
[tinychat-tutorial](https://github.com/mit-han-lab/tinychat-tutorial) in plain PyTorch:

- the **LLaMA forward pass is written by hand** in [tinychat.py](tinychat.py) - RMSNorm, RoPE,
  grouped-query attention with a KV cache, SwiGLU - no `transformers` model code;
- every linear layer can be stored in **fp32, bf16, INT8** (symmetric, per output channel) or
  **INT4** (affine, per group of 64) - the scheme from
  [01_quantization](../../examples/01_quantization.ipynb) - and multiplied with PyTorch's
  integer-weight CPU kernels;
- the chat reuses the KV cache across turns, so each turn only processes the new tokens.

Default model: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
(LLaMA architecture, ~720 MB download on first run). Any LLaMA-architecture repo or local folder
works via `--model`.

## Run

```bash
pip install -r requirements.txt
python tinychat.py                    # chat, bf16 weights
python tinychat.py --precision int4   # chat with 4-bit weights
python tinychat.py --precision awq4   # 4-bit weights after AWQ scaling
python tinychat.py --bench            # size / perplexity / speed of every precision
python test_tinychat.py               # self-check vs transformers' LlamaForCausalLM (needs transformers)
```

In the chat, `/reset` clears the conversation and an empty line quits.

## Results

`python tinychat.py --bench` on an Intel i5-11400H laptop (6 cores, 16 GB RAM), SmolLM2-360M-Instruct,
decoding 64 tokens greedily; perplexity is on a short built-in paragraph:

| precision | weights | perplexity | decode |
|-----------|--------:|-----------:|-------:|
| fp32      | 1447 MB | 33.19 | 15.4 tok/s |
| **bf16**  |  724 MB | 33.22 | **20.1 tok/s** |
| int8      |  363 MB | 33.22 |  6.0 tok/s |
| int4      |  244 MB | 40.58 |  2.3 tok/s |
| **awq4**  |  244 MB | **34.84** |  2.9 tok/s |

What it shows:

- **Size is the guaranteed win.** INT8 is 4x smaller than fp32 at the same perplexity; INT4 is ~6x
  smaller. INT4 keeps the output head (`lm_head`) in INT8: it is the most sensitive layer, and
  quantizing it to 4 bits alone raised perplexity from 40.6 to 45.3. Plain round-to-nearest INT4
  still costs ~22% perplexity on a model this small.
- **AWQ closes most of that gap for free.** `awq4` (as used by TinyChatEngine) scales each input
  channel by `s = mean|x|^a` before INT4 rounding, picking `a` per layer to minimize the INT4
  output error on a short calibration paragraph, and folds `1/s` into the preceding RMSNorm (or
  `up_proj`), so the float model is unchanged. Same 244 MB, perplexity +5% instead of +22%.
  The eval paragraph is short (~100 tokens), so treat perplexity differences of ~1 as noise.
- **Speed depends on the kernel, not just the bit width.** Decoding reads every weight once per
  token, so it is memory-bound: bf16 halves the bytes and is ~1.3x faster than fp32. INT8/INT4 read
  even fewer bytes, but PyTorch's generic integer-weight CPU kernels are slower than the MKL float
  GEMM on this CPU, so they lose. TinyChatEngine wins here with hand-written SIMD + multithreaded
  INT4 kernels - exactly what [05_kernel_optimization](../../examples/05_kernel_optimization.ipynb)
  is about.
