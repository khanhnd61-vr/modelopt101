"""Self-check, no download needed: python test_tinychat.py"""
import torch

from tinychat import Linear, Llama, awq


def test_quantized_linear():
    torch.manual_seed(0)
    for N, K in [(320, 960), (2560, 960), (960, 2560)]:
        w, x = torch.randn(N, K) * 0.05, torch.randn(3, K)
        ref = x @ w.T
        for p, tol in [("fp32", 1e-4), ("bf16", 2e-2), ("int8", 2e-2), ("int4", 2e-1)]:
            err = (Linear(w, p)(x) - ref).abs().max() / ref.abs().max()
            assert err < tol, (p, N, K, err)
    assert Linear(w, "int4").nbytes() < Linear(w, "int8").nbytes() < Linear(w, "fp32").nbytes() / 3


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = dict(vocab_size=97, hidden_size=128, intermediate_size=192, num_hidden_layers=2,
               num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, rope_theta=10000.0,
               max_position_embeddings=64, tie_word_embeddings=True)
    torch.manual_seed(0)
    hf = LlamaForCausalLM(LlamaConfig(**cfg, attn_implementation="eager")).eval()
    return cfg, hf


def test_matches_transformers_and_kv_cache():
    cfg, hf = tiny_llama()
    ours = Llama(cfg, hf.state_dict(), "fp32", max_ctx=64)
    ids = torch.randint(0, cfg["vocab_size"], (20,))
    with torch.no_grad():
        ref = hf(ids[None]).logits[0]
    full = ours.forward(ids, 0, all_logits=True)
    assert (full - ref).abs().max() < 1e-4, (full - ref).abs().max()

    # prefill 12 tokens, then feed the rest one at a time through the KV cache
    ours.forward(ids[:12], 0)
    for t in range(12, 20):
        step = ours.forward(ids[t:t + 1], t)
    assert (step[0] - ref[-1]).abs().max() < 1e-4


def test_awq_keeps_float_model():
    # AWQ only moves scale between a linear and what feeds it: the float model must not change
    cfg, hf = tiny_llama()
    sd = hf.state_dict()
    for k, v in sd.items():           # uneven channel magnitudes, as in real models, so AWQ picks a > 0
        if "layernorm" in k:
            v.uniform_(0.1, 5.0)
    ids = torch.randint(0, cfg["vocab_size"], (24,))
    scaled = awq(cfg, sd, ids.tolist())
    before = Llama(cfg, sd, "fp32", max_ctx=64).forward(ids, 0, all_logits=True)
    after = Llama(cfg, scaled, "fp32", max_ctx=64).forward(ids, 0, all_logits=True)
    assert (after - before).abs().max() / before.abs().max() < 1e-5, (after - before).abs().max()
    assert any((scaled[k] != v).any() for k, v in sd.items()), "no layer was rescaled"


if __name__ == "__main__":
    test_quantized_linear()
    test_matches_transformers_and_kv_cache()
    test_awq_keeps_float_model()
    print("all checks passed")
