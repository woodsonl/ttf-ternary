"""A converted Ling-3.0 run (ttfconv.bailing.convert's OUT) evaluated or sampled layer-major, one decoder layer on
the device at a time (ttf.bailing.student).

eval: the held-out windows of the conversion (meta.json: the last eval_windows * seq tokens of DATA) through every
converted layer; prints the mean next-token loss with convert.eval_loss's head (final norm, the rotation undone, the
untied head). The embedding enters as emb @ Q^T when the run is folded.
--prompt TEXT: greedy generation of --new tokens (the whole sequence recomputed per token, no cache).
--act-bits 0: no activation quantisation, the converter's own fake-quantised layer (its eval.json "ternary");
8 (default): per-token int8 activations, as the TTF kernels run. --adapters DIR: float adapters from
ttf.bailing.train (DIR/A{i}.pt). CUDA runs in bf16, CPU in fp32 (as the converter).
  python -m ttf.bailing.eval HF_DIR OUT DATA.bin [--act-bits 8] [--adapters DIR] [--device cuda] [--mb 4]
  python -m ttf.bailing.eval HF_DIR OUT --prompt "The capital of France is" [--new 24]
"""
import argparse, json, os, time
import numpy as np
import torch

from ttf.bailing.student import student, offload, run
from ttfconv.bailing import convert as C
from ttfconv.fold import residual_q


class Run:
    """a converted run: config, Hub reader, meta, rotation, embedding"""
    def __init__(self, hf, out, dev):
        self.cfg, self.rd, self.out, self.dev = C.load_config(hf), C.Reader(hf), out, dev
        self.meta = json.load(open(os.path.join(out, "meta.json")))
        self.dt = torch.bfloat16 if dev.startswith("cuda") else torch.float32
        self.Q = None if self.meta["no_fold"] else residual_q(self.cfg.hidden_size,
                                                              torch.Generator().manual_seed(self.meta["seed"]))
        self.emb = self.rd.get("model.embed_tokens.weight", "model.word_embeddings.weight")

    def held_out(self, data):
        """the conversion's held-out windows [eval_windows, seq]"""
        d = np.memmap(data, dtype=np.uint32, mode="r")
        s, n = self.meta["seq"], self.meta["eval_windows"]
        return torch.from_numpy(np.stack([d[a:a + s].astype(np.int64) for a in range(len(d) - n * s, len(d), s)]))

    def embed(self, ids):
        e = self.emb[ids].float()
        return (e @ self.Q.T if self.Q is not None else e).to(self.dt)

    def layer(self, li, act_bits=8, adapters=None):
        blob = torch.load(os.path.join(self.out, f"L{li}.pt"), map_location="cpu", weights_only=True)
        ad = None
        if adapters:
            ad = torch.load(os.path.join(adapters, f"A{li}.pt"), map_location="cpu", weights_only=True)
        r = next((int(v.shape[-1]) for k, v in (ad or {}).items() if k.endswith(".A")), 0)
        m = student(self.cfg, li, blob, self.dt, lora=r, act_bits=act_bits)
        if ad:
            miss, unexp = m.load_state_dict(ad, strict=False)
            assert not unexp and not [k for k in miss if ".lora" in k], (li, unexp)
        return m

    def loss(self, hs, ids, mb=4):
        return C.eval_loss(self.rd, self.cfg, hs, ids, self.Q, self.dev, mb)

    def logits(self, h):
        """the head on residual rows h [n, d] (convert.eval_loss's arithmetic)"""
        if not hasattr(self, "_head"):
            self._head = (self.rd.get("model.norm.weight").float().to(self.dev),
                          self.rd.get("lm_head.weight").float().to(self.dev))
        g, H = self._head
        x = h.float().to(self.dev)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        if self.Q is not None:
            x = x @ self.Q.to(self.dev)
        return (x * g) @ H.T


@torch.no_grad()
def stream(layers, hs, dev, mb=4, log=None):
    """hs [n, T, d] (host) through each module of `layers` (host-resident, moved to dev one at a time), in place"""
    for li, m in enumerate(layers):
        t0 = time.time()
        m.to(dev)
        for b0 in range(0, hs.shape[0], mb):
            hs[b0:b0 + mb] = run(m, hs[b0:b0 + mb].to(dev)).to("cpu", hs.dtype)
        offload(m)
        if log:
            log(f"layer {li}: {time.time() - t0:.1f}s, h rms {float(hs.float().pow(2).mean().sqrt()):.3f}")
    return hs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("hf"); ap.add_argument("out"); ap.add_argument("data", nargs="?")
    ap.add_argument("--act-bits", type=int, default=8); ap.add_argument("--adapters")
    ap.add_argument("--mb", type=int, default=4); ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--prompt"); ap.add_argument("--new", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
    R = Run(a.hf, a.out, a.device)
    n = a.layers or R.cfg.num_hidden_layers
    if a.prompt is None:
        ids = R.held_out(a.data)
        hs = stream((R.layer(i, a.act_bits, a.adapters) for i in range(n)), R.embed(ids), a.device, a.mb, log)
        loss = R.loss(hs, ids, a.mb)
        print(json.dumps({"held_out_loss": loss, "act_bits": a.act_bits, "adapters": a.adapters,
                          "windows": list(ids.shape), "layers": n}))
        return loss
    from transformers import PreTrainedTokenizerFast      # (AutoTokenizer reads config.json, which needs load_config)
    tok = PreTrainedTokenizerFast.from_pretrained(a.hf)
    ids = tok(a.prompt, return_tensors="pt").input_ids
    layers = [R.layer(i, a.act_bits, a.adapters) for i in range(n)]
    for _ in range(a.new):
        hs = stream(layers, R.embed(ids), a.device)
        nxt = int(R.logits(hs[0, -1:]).argmax(-1))
        ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
        if nxt == R.cfg.eos_token_id:
            break
    text = tok.decode(ids[0])
    print(text)
    return text


if __name__ == "__main__":
    main()
