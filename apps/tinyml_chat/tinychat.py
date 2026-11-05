"""TinyChat - a from-scratch LLaMA chat app that runs on a laptop CPU.

The idea of TinyChatEngine in plain PyTorch: the LLaMA forward pass is written out by hand
(RMSNorm, RoPE, grouped-query attention with a KV cache, SwiGLU), and every linear layer can be
stored in fp32, bf16, INT8 (per output channel) or INT4 (per group) - the scheme from
examples/01_quantization - and multiplied with PyTorch's integer-weight CPU kernels.

    python tinychat.py                    # chat (bf16)
    python tinychat.py --precision int4   # chat with 4-bit weights
    python tinychat.py --bench            # size / perplexity / speed for every precision
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
PRECISIONS = ("fp32", "bf16", "int8", "int4", "awq4")   # awq4 = int4 after AWQ scaling


# ---------------------------------------------------------------------------- quantized linear

def int4_groups(w):
    """Affine INT4, one (scale, zero) per group of K: w ~= (q - 8) * scale + zero, q in [0, 15]."""
    N, K = w.shape
    group = 64 if K % 64 == 0 else 32
    g = w.reshape(N, K // group, group)
    lo, hi = g.amin(-1), g.amax(-1)
    scale = ((hi - lo) / 15).clamp(min=1e-8)
    q = torch.round((g - lo[..., None]) / scale[..., None]).clamp(0, 15)
    return q, scale, lo, group


def fake_int4(w):
    """w after an INT4 round trip, in float."""
    q, scale, lo, _ = int4_groups(w)
    return (q * scale[..., None] + lo[..., None]).reshape(w.shape)


class Linear:
    """y = x @ W.T with W stored at the given precision."""

    def __init__(self, w, precision):
        w = w.float()
        self.precision = precision
        if precision == "fp32":
            self.w = w.contiguous()
        elif precision == "bf16":
            self.w = w.to(torch.bfloat16).contiguous()
        elif precision == "int8":
            # symmetric, one scale per output channel: q = round(w / s), s = max|w| / 127
            self.s = (w.abs().amax(1) / 127).clamp(min=1e-8)
            self.q = torch.round(w / self.s[:, None]).clamp(-128, 127).to(torch.int8)
        elif precision == "int4":
            q, scale, lo, self.group = int4_groups(w)
            self.q = torch.ops.aten._convert_weight_to_int4pack_for_cpu(q.reshape(w.shape).to(torch.int32), 1)
            self.sz = torch.stack([scale, lo + 8 * scale], -1).transpose(0, 1).contiguous()
        else:
            raise ValueError(f"precision must be one of {PRECISIONS}")

    def __call__(self, x):
        if self.precision == "fp32":
            return x @ self.w.T
        if self.precision == "bf16":
            return (x.to(torch.bfloat16) @ self.w.T).float()
        if self.precision == "int8":
            return torch.ops.aten._weight_int8pack_mm(x, self.q, self.s)
        return torch.ops.aten._weight_int4pack_mm_for_cpu(x, self.q, self.group, self.sz)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for k, t in vars(self).items() if torch.is_tensor(t))


# ---------------------------------------------------------------------------- the model

def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


class Llama:
    def __init__(self, cfg, weights, precision="fp32", max_ctx=2048):
        self.cfg = cfg
        self.eps = cfg["rms_norm_eps"]
        self.n_heads, self.n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or cfg["hidden_size"] // self.n_heads
        W = lambda name: weights[name].float()

        self.embed = W("model.embed_tokens.weight")
        self.norm = W("model.norm.weight")
        head = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])   # tied embeddings
        # the output head is the most sensitive layer: at int4 it alone costs ~+12% perplexity, so keep it int8
        self.lm_head = Linear(head, "int8" if precision == "int4" else precision)
        self.layers = []
        for i in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{i}."
            self.layers.append({
                "ln1": W(p + "input_layernorm.weight"), "ln2": W(p + "post_attention_layernorm.weight"),
                **{n: Linear(weights[p + f"self_attn.{n}_proj.weight"], precision) for n in "qkvo"},
                **{n: Linear(weights[p + f"mlp.{n}_proj.weight"], precision) for n in ("gate", "up", "down")},
            })

        # RoPE tables (HF "rotate_half" layout) and the KV cache
        inv_freq = 1.0 / cfg.get("rope_theta", 10000.0) ** (torch.arange(0, self.head_dim, 2) / self.head_dim)
        freqs = torch.outer(torch.arange(max_ctx).float(), inv_freq)
        self.cos, self.sin = torch.cat((freqs, freqs), -1).cos(), torch.cat((freqs, freqs), -1).sin()
        self.max_ctx = max_ctx
        shape = (len(self.layers), self.n_kv, max_ctx, self.head_dim)
        self.k_cache, self.v_cache = torch.zeros(shape), torch.zeros(shape)

    @classmethod
    def load(cls, path, precision="fp32", max_ctx=2048):
        from safetensors.torch import load_file
        cfg = json.loads((path / "config.json").read_text())
        weights = {}
        for f in sorted(path.glob("*.safetensors")):
            weights.update(load_file(f))
        if precision == "awq4":
            from tokenizers import Tokenizer
            ids = Tokenizer.from_file(str(path / "tokenizer.json")).encode(CALIB_TEXT, add_special_tokens=False).ids
            weights, precision = awq(cfg, weights, ids), "int4"
        return cls(cfg, weights, precision, max_ctx)

    def weight_bytes(self):
        return sum(lin.nbytes() for L in self.layers for lin in L.values() if isinstance(lin, Linear)) \
            + self.lm_head.nbytes()

    @torch.no_grad()
    def forward(self, ids, start, all_logits=False):
        """Run tokens `ids` that sit at positions start..start+T-1; earlier ones are in the KV cache."""
        T, end = len(ids), start + len(ids)
        assert end <= self.max_ctx, "context full"
        x = self.embed[ids]
        cos, sin = self.cos[start:end, None], self.sin[start:end, None]
        mask = torch.arange(end)[None, :] <= torch.arange(start, end)[:, None]       # causal, with cache
        for i, L in enumerate(self.layers):
            h = rms_norm(x, L["ln1"], self.eps)
            q = L["q"](h).view(T, self.n_heads, self.head_dim)
            k = L["k"](h).view(T, self.n_kv, self.head_dim)
            v = L["v"](h).view(T, self.n_kv, self.head_dim)
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            self.k_cache[i, :, start:end] = k.transpose(0, 1)
            self.v_cache[i, :, start:end] = v.transpose(0, 1)
            a = F.scaled_dot_product_attention(q.transpose(0, 1), self.k_cache[i, :, :end],
                                               self.v_cache[i, :, :end], attn_mask=mask, enable_gqa=True)
            x = x + L["o"](a.transpose(0, 1).reshape(T, -1))
            h = rms_norm(x, L["ln2"], self.eps)
            x = x + L["down"](F.silu(L["gate"](h)) * L["up"](h))
        x = rms_norm(x if all_logits else x[-1:], self.norm, self.eps)
        return self.lm_head(x)


# ---------------------------------------------------------------------------- AWQ

# calibration text for AWQ - deliberately not EVAL_TEXT, which perplexity is measured on
CALIB_TEXT = (
    "The bicycle was invented in the early nineteenth century and changed how people moved through "
    "cities. Early models had no pedals; riders pushed against the ground with their feet. Chains, "
    "air-filled tyres and lighter steel frames followed, and by 1900 millions of workers rode to "
    "factories every day. Today engineers still argue about the best frame geometry, while city "
    "planners build protected lanes so that children and older riders feel safe. A good bicycle "
    "converts almost all of the rider's effort into forward motion, which makes it one of the most "
    "efficient machines ever built. Recipes, weather reports, football results and short stories "
    "all appear in the same newspaper, written for readers with very different interests."
)


@torch.no_grad()
def awq(cfg, weights, calib_ids, grid=20):
    """Activation-aware weight scaling before INT4 (AWQ, Lin et al. 2023), on a copy of `weights`.

    For each linear group, input channel j is scaled: W[:, j] *= s[j] and x[j] /= s[j], with
    s = mean|x|^a and a in [0, 1) picked to minimize the INT4 output error on calibration tokens
    (a = 0 is plain round-to-nearest, so it never does worse there). The 1/s is folded into the
    RMSNorm weight before q,k,v and gate,up, and into up_proj's rows before down_proj, so the
    float model computes exactly the same function.
    """
    w = {k: v.float().clone() for k, v in weights.items()}
    model = Llama(cfg, w, "fp32", max_ctx=len(calib_ids))
    inputs = {}

    def record(lin, key):
        def call(x):
            inputs[key] = x
            return lin(x)
        return call

    for i, L in enumerate(model.layers):
        for n in ("q", "gate", "down"):
            L[n] = record(L[n], (i, n))
    model.forward(torch.tensor(calib_ids), 0, all_logits=True)
    del model

    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        groups = [   # (recorded input, linears sharing it, where 1/s is folded)
            ("q", [f"{p}self_attn.{n}_proj.weight" for n in "qkv"], p + "input_layernorm.weight"),
            ("gate", [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"], p + "post_attention_layernorm.weight"),
            ("down", [p + "mlp.down_proj.weight"], p + "mlp.up_proj.weight"),
        ]
        for key, names, fold in groups:
            x = inputs[(i, key)]
            W = torch.cat([w[n] for n in names])
            ref, mean = x @ W.T, x.abs().mean(0)
            best_err, best_s = float("inf"), None
            for a in (torch.arange(grid) / grid).tolist():
                s = mean.pow(a).clamp(min=1e-4)
                s = s / (s.max() * s.min()).sqrt()
                err = ((x / s) @ fake_int4(W * s).T - ref).pow(2).mean().item()
                if err < best_err:
                    best_err, best_s = err, s
            for n in names:
                w[n] *= best_s
            w[fold] /= best_s if w[fold].dim() == 1 else best_s[:, None]
    return w


# ---------------------------------------------------------------------------- chat

class Chat:
    def __init__(self, path, precision, max_ctx=2048):
        from jinja2 import Template
        from tokenizers import Tokenizer
        self.model = Llama.load(path, precision, max_ctx)
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        tc = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        token = lambda t: t["content"] if isinstance(t, dict) else (t or "")
        self.specials = {"bos_token": token(tc.get("bos_token")), "eos_token": token(tc.get("eos_token"))}
        self.template = Template(tc["chat_template"])
        gen = path / "generation_config.json"
        eos = json.loads(gen.read_text()).get("eos_token_id") if gen.exists() else None
        self.stop = {self.tok.token_to_id(self.specials["eos_token"])} | set(eos if isinstance(eos, list) else [eos])
        self.cached = []          # token ids currently held in the KV cache

    def prompt_ids(self, messages):
        text = self.template.render(messages=messages, add_generation_prompt=True, **self.specials)
        return self.tok.encode(text, add_special_tokens=False).ids

    def generate(self, ids, max_new=256, temperature=0.7, top_p=0.9, stream=None):
        """Generate after `ids`, reusing the KV cache for the prefix it shares with the last call."""
        n = 0
        while n < min(len(self.cached), len(ids) - 1) and self.cached[n] == ids[n]:
            n += 1
        if len(ids) + max_new > self.model.max_ctx:
            n = 0
            ids = ids[-(self.model.max_ctx - max_new):]
        logits = self.model.forward(torch.tensor(ids[n:]), n)
        out = []
        for _ in range(max_new):
            nxt = sample(logits[-1], temperature, top_p)
            if nxt in self.stop:
                break
            out.append(nxt)
            if stream:
                stream(out)
            logits = self.model.forward(torch.tensor([nxt]), len(ids) + len(out) - 1)
        self.cached = ids + out
        return out


def sample(logits, temperature, top_p):
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, -1)
    p, idx = probs.sort(descending=True)
    keep = p.cumsum(0) - p < top_p                     # smallest set whose mass reaches top_p
    return int(idx[keep][torch.multinomial(p[keep], 1)])


# ---------------------------------------------------------------------------- benchmark

EVAL_TEXT = (
    "A laptop has far less memory bandwidth than a data-center GPU, so running a language model "
    "on it is mostly a question of how many bytes must be read for every generated token. Each "
    "new token needs the whole set of weights once. Storing those weights in fewer bits shrinks "
    "the model and the traffic, while the quality of its predictions should change as little as "
    "possible. Perplexity measures that quality: it is the exponential of the average negative "
    "log-likelihood the model assigns to the next token of a held-out text, so lower is better."
)


def perplexity(model, ids):
    logits = model.forward(torch.tensor(ids), 0, all_logits=True)
    return math.exp(F.cross_entropy(logits[:-1], torch.tensor(ids[1:])).item())


def bench(path, precisions, n_tokens=64):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    eval_ids = tok.encode(EVAL_TEXT, add_special_tokens=False).ids
    rows = []
    for p in precisions:
        chat = Chat(path, p)
        ppl = perplexity(chat.model, eval_ids)
        ids = chat.prompt_ids([{"role": "user", "content": "Explain what model quantization is in two sentences."}])
        chat.generate(ids, max_new=4, temperature=0)               # warm-up
        chat.cached = []
        t = time.perf_counter()
        out = chat.generate(ids, max_new=n_tokens, temperature=0)
        tps = len(out) / (time.perf_counter() - t)
        rows.append((p, chat.model.weight_bytes() / 1e6, ppl, tps, chat.tok.decode(out)))
        print(f"  {p}: done")
    print(f"\n{'precision':<10}{'weights':>11}{'perplexity':>12}{'decode':>14}")
    print("-" * 47)
    for p, mb, ppl, tps, _ in rows:
        print(f"{p:<10}{mb:>9.0f}MB{ppl:>12.2f}{tps:>10.1f} tok/s")
    for p, *_, text in rows:
        print(f"\n[{p}] {text.strip()[:300]}")


# ---------------------------------------------------------------------------- CLI

def download(model):
    if Path(model).is_dir():
        return Path(model)
    try:                                   # trust the OS certificate store (TLS-inspecting networks)
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local folder (LLaMA architecture)")
    ap.add_argument("--precision", default="bf16", choices=PRECISIONS)
    ap.add_argument("--bench", action="store_true", help="compare all precisions and exit")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    path = download(args.model)
    if args.bench:
        return bench(path, PRECISIONS)

    t = time.perf_counter()
    chat = Chat(path, args.precision)
    print(f"{args.model} [{args.precision}, {chat.model.weight_bytes() / 1e6:.0f}MB weights] "
          f"loaded in {time.perf_counter() - t:.1f}s. Empty line or Ctrl+C to quit, /reset to clear.\n")
    messages = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        if user == "/reset":
            messages, chat.cached = [], []
            continue
        messages.append({"role": "user", "content": user})
        print("Bot: ", end="", flush=True)
        shown = [""]

        def stream(out):
            text = chat.tok.decode(out)
            print(text[len(shown[0]):], end="", flush=True)
            shown[0] = text

        t = time.perf_counter()
        out = chat.generate(chat.prompt_ids(messages), args.max_new, args.temperature, stream=stream)
        print(f"\n  ({len(out) / (time.perf_counter() - t):.1f} tok/s)\n")
        messages.append({"role": "assistant", "content": chat.tok.decode(out)})


if __name__ == "__main__":
    main()
