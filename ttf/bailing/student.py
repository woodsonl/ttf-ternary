"""One converted Ling-3.0 (HF model_type bailing_hybrid) decoder layer as a runnable module, and its float teacher.

Needs ttf-conversion on the path: the config, the Hub reader, the one-layer shell and the fold are
ttfconv.bailing.convert's (load_config, Reader, shell, fold_rules / fold_state).

student(cfg, li, blob): transformers' one-layer shell with every matrix of a converted layer file (OUT/L{i}.pt from
ttfconv.bailing.convert) as a TernLinear, the routed experts' gate+up and down as two TernaryExperts run sorted by
expert (convert.experts_forward's order), and the layer's remaining tensors from blob["_float"]; the swaps follow the
file's keys and everything loads strictly. lora > 0: a float adapter A B^T (A [E, N, r] at 0, B [E, r, K]) beside
each ternary matrix, on its input; the adapters are the module's only trainable parameters.
teacher(rd, cfg, li, rules, Q): the same shell with the Hub's float layer, folded as the conversion folded it.
run(m, x): the layer on inputs_embeds x [B, T, d] -> [B, T, d].
"""
import torch
import torch.nn as nn

from ttf.q38.moe import TernaryExperts, TernLinear, _Pad
from ttfconv.bailing import convert as C

EXPERTS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


class Lora(nn.Module):
    def __init__(self, E, K, N, r):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(E, N, r))
        self.B = nn.Parameter(torch.randn(E, r, K) / K ** 0.5)

    def forward(self, x, offs=None):
        A, B = self.A.to(x.dtype), self.B.to(x.dtype)
        if offs is None:
            return x @ B[0].T @ A[0].T
        pd = _Pad(offs, A.shape[0], x.device)
        return pd.unpad(torch.bmm(torch.bmm(pd.pad(x), B.transpose(1, 2)), A.transpose(1, 2)))


class Adapted(nn.Module):
    """a TernLinear plus its float adapter"""
    def __init__(self, t, r):
        super().__init__()
        self.t, self.lora = t, Lora(1, t.K, t.N, r)

    def forward(self, x):
        return self.t(x) + self.lora(x)


class Experts(nn.Module):
    """BailingHybridExperts' call (x [T, d], top-k index and weights [T, k]) on ternary gate+up and down"""
    def __init__(self, act_fn, gate_up_proj, down_proj, r=0):
        super().__init__()
        self.act_fn, self.gate_up_proj, self.down_proj = act_fn, gate_up_proj, down_proj
        self.E = gate_up_proj.E
        self.lora_gu = Lora(self.E, gate_up_proj.K, gate_up_proj.N, r) if r else None
        self.lora_dn = Lora(self.E, down_proj.K, down_proj.N, r) if r else None

    def forward(self, x, idx, w):
        T, k = idx.shape
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = torch.arange(T, device=x.device).repeat_interleave(k)[order]
        offs = torch.zeros(self.E + 1, dtype=torch.long, device=x.device)
        offs[1:] = torch.bincount(flat, minlength=self.E).cumsum(0)
        xs = x[tok]
        h = self.gate_up_proj(xs, offs)
        if self.lora_gu is not None:
            h = h + self.lora_gu(xs, offs)
        gate, up = h.chunk(2, -1)
        mid = self.act_fn(gate) * up
        o = self.down_proj(mid, offs)
        if self.lora_dn is not None:
            o = o + self.lora_dn(mid, offs)
        o = o * w.reshape(-1)[order, None].to(o.dtype)
        return torch.zeros_like(x).index_add(0, tok, o.to(x.dtype))


def _set(L, path, mod):
    parent, _, leaf = path.rpartition(".")
    setattr(L.get_submodule(parent), leaf, mod)


def student(cfg, li, blob, dt=torch.float32, lora=0, act_bits=8):
    """blob: torch.load(OUT/L{li}.pt). The shell's float tensors in dt; the ternary modules' scales in fp32."""
    m = C.shell(cfg, li)
    L = m.layers[0]
    tern = {}
    for name, st in blob.items():
        if name == "_float":
            continue
        E, N = st["row_scale"].shape
        K = st["col_scale"].shape[1]
        kw = dict(act_bits=act_bits, centring="in_mean" in st,
                  adapter_rank=int(st["ad_cb"].shape[1]) if "ad_cb" in st else 0)
        t = TernaryExperts(E, K, N, **kw) if name in EXPERTS else TernLinear(K, N, **kw)
        t.load_state_dict(st, strict=True)
        t.requires_grad_(False)
        tern[name] = t
    miss, unexp = L.load_state_dict(blob["_float"], strict=False)
    assert not unexp, f"L{li}: unexpected {unexp[:5]}"
    left = [k for k in miss if not any(k == n or k.startswith(n + ".") for n in tern)]
    assert not left, f"L{li}: missing {left[:5]}"
    m.to(dt).requires_grad_(False)
    if EXPERTS[0] in tern:
        L.mlp.experts = Experts(L.mlp.experts.act_fn, tern.pop(EXPERTS[0]), tern.pop(EXPERTS[1]), lora)
    for name, t in tern.items():
        _set(L, name, Adapted(t, lora) if lora else t)
    return m.eval()


def teacher(rd, cfg, li, rules=None, Q=None, dev="cpu", dt=torch.float32):
    """the Hub's float layer li, folded (rules, Q) as ttfconv.bailing.convert folds it (dev: where the fold runs)"""
    sd = rd.layer(li)
    if rules is not None:
        C.fold_state(sd, rules[(cfg.layer_types[li], cfg.mlp_layer_types[li])], Q, dev)
    m = C.shell(cfg, li)
    m.layers[0].load_state_dict(sd, strict=True, assign=True)
    return m.to(dt).requires_grad_(False).eval()


def adapters(m):
    """the student's float adapter parameters {name: tensor}"""
    return {n: p for n, p in m.named_parameters() if p.requires_grad}


def offload(m):
    """m back to the host, the TernaryExperts' unpacked-trit caches dropped (they would keep device memory)"""
    m.to("cpu")
    for t in m.modules():
        if isinstance(t, TernaryExperts):
            t.invalidate()
    return m


def run(m, x):
    return m(inputs_embeds=x, use_cache=False).last_hidden_state
