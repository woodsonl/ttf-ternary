"""ttf.bailing on a tiny random bailing_hybrid converted by ttfconv.bailing.convert (CPU, fp32; needs ttf-conversion on
the path). Stage 1: the layer-major student without activation quantisation reproduces the converter's own held-out
ternary loss. Stage 2: the adapter trainer's step 0 is that loss (the adapters start at zero), its distillation loss
falls, and the trained adapters bring the students' held-out stream closer to the teachers' (a random tiny model's
token loss sits at log(vocab) for both, so the stream is the measure here); the evaluator reads them back."""
import json
import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
C = pytest.importorskip("ttfconv.bailing.convert")
from transformers import CONFIG_MAPPING
from ttfconv.arch import hf_class

from ttf.bailing import eval as E, train as TR

SEQ, EVAL = 32, 2


@pytest.fixture(scope="module")
def conv(tmp_path_factory):
    if "bailing_hybrid" not in CONFIG_MAPPING:
        pytest.skip("transformers has no bailing_hybrid")
    from safetensors.torch import save_file
    d = tmp_path_factory.mktemp("bailing")
    json.dump(dict(hidden_size=128, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                   num_key_value_heads=4, head_dim=32, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
                   kv_lora_rank=32, layer_group_size=3, first_k_dense_replace=1, moe_intermediate_size=64,
                   num_local_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1, vocab_size=512,
                   max_position_embeddings=512, pad_token_id=0, eos_token_id=1, kda_lower_bound=-5),
              open(d / "config.json", "w"))
    cfg = C.load_config(str(d))
    torch.manual_seed(0)
    m = hf_class("bailing_hybrid")(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("layernorm.weight") or n == "model.norm.weight":
                p.normal_(1.0, 0.2)
    save_file(m.state_dict(), str(d / "model.safetensors"))
    np.random.default_rng(0).integers(2, 512, 8192, dtype=np.uint32).tofile(d / "data.bin")
    out = d / "out"
    C.main([str(d), str(d / "data.bin"), str(out), "--windows", "4", "--seq", str(SEQ), "--eval-windows", str(EVAL),
            "--mb", "2", "--rank", "4", "--chunk", "2", "--experts-pass", "2", "--device", "cpu"])
    return d, out, json.load(open(out / "eval.json"))


def test_student_parity(conv):
    d, out, ev = conv
    loss = E.main([str(d), str(out), str(d / "data.bin"), "--act-bits", "0", "--device", "cpu"])
    assert abs(loss - ev["ternary"]) < 1e-4 * ev["ternary"], (loss, ev)
    q = E.main([str(d), str(out), str(d / "data.bin"), "--act-bits", "8", "--device", "cpu"])
    assert abs(q - ev["ternary"]) < 0.05 * ev["ternary"], (q, ev)        # int8 activations: close, not equal


def test_adapter_training(conv, tmp_path):
    d, out, ev = conv
    curve = TR.main([str(d), str(out), str(d / "data.bin"), str(tmp_path), "--steps", "12", "--batch", "4",
                     "--eval-every", "12", "--lr", "1e-2", "--min-lr", "1e-3", "--warmup", "2", "--rank", "4",
                     "--device", "cpu"])
    held = [c["held_out"] for c in curve if "held_out" in c]
    kd = [c["kd"] for c in curve if "kd" in c]
    assert abs(held[0] - ev["ternary"]) < 1e-4 * ev["ternary"], (held, ev)
    assert np.mean(kd[-4:]) < np.mean(kd[:4]), kd
    R = E.Run(str(d), str(out), "cpu")
    ids = R.held_out(str(d / "data.bin"))
    n = R.cfg.num_hidden_layers
    t = E.stream([TR.teacher(R.rd, R.cfg, i, C.fold_rules(R.cfg), R.Q) for i in range(n)], R.embed(ids), "cpu")
    err = lambda ad: float((E.stream((R.layer(i, 0, ad) for i in range(n)), R.embed(ids), "cpu") - t).norm() / t.norm())
    assert err(str(tmp_path)) < err(None)
    again = E.main([str(d), str(out), str(d / "data.bin"), "--act-bits", "0", "--adapters", str(tmp_path),
                    "--device", "cpu"])
    assert abs(again - held[-1]) < 1e-4 * held[-1], (again, held)
