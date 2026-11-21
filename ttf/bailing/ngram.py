"""N-gram recovery table for a converted Ling-3.0 run (ttfconv.bailing.convert's OUT): every weight frozen, a hashed
n-gram embedding added to the residual stream, trained by distillation from the float teacher.

Table at injection layer L (0: added to the embedding; num_hidden_layers: to the final stream before the head): two
blocks of rows/2 x d, T2 and T3, zero to start (the model unchanged). Position t of a window reads
T2[hash(x[t-1], x[t])] + T3[hash(x[t-2], x[t-1], x[t])] (n-grams inside the window; position 0 reads nothing,
position 1 only T2) and adds it to the stream entering layer L. Tables, lookups and optimizer state live on the host.

Training (--steps > 0), layer-major: the training windows' teacher logits come from one teacher pass (its final stream
is cached), the student's stream entering the first injection layer from one student pass. A step takes --batch
windows (an epoch is one pass over the training windows in a fresh order) forward from that cache through the student
layers, the tables added; the loss is KL(teacher || student) per token at the head; the gradient goes back layer by
layer (each layer recomputed from its saved input), and at each injection layer the rows it touched take one lazy Adam
step. --steps 0: the closed-form ablation instead, matching the streams at the injection layer: D = teacher stream -
student stream there, T2 the ridge mean of D per 2-gram bucket (sum / (count + lam)), T3 the same on D - T2[h2].

Eval: mean next-token loss (convert.eval_loss) of the teacher, the student and the student with each trained table on
three sets: the conversion's held-out windows (the tail of --conv-data), --eval-windows windows from the tail of DATA
held out of training (--hold tokens), and as many of the training windows. Coverage per set and order: "seen", the share
of positions whose n-gram occurs in the training windows; "hit", the share whose bucket a training n-gram touched (a
collision reads another n-gram's row).
  python -m ttf.bailing.ngram HF_DIR CONV DATA.bin --conv-data CONV_DATA.bin --tokens 819200 --steps 200 \
      --cfg 2/2000000 --cfg 2+8+14+20/500000 [--adapters DIR | --student-hf DIR] [--lr 1e-2]
CONV supplies the held-out windows, the rotation seed and (without --student-hf) the ternary student.
--cfg LAYERS/ROWS[/LAM]: injection layers joined by +, rows per injection layer, ridge lam (--steps 0 only); each
config prints one line of JSON.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from ttf.bailing.eval import Run, stream
from ttf.bailing.student import teacher, offload, run
from ttf.q38.train import lr_at
from ttfconv.bailing import convert as C

MIX = {2: 0x9E3779B97F4A7C15 - 2 ** 64, 3: 0xC2B2AE3D27D4EB4F - 2 ** 64}    # odd int64 multipliers, one per order


def keys(ids, n, V):
    """the n-gram ending at each position t >= n - 1 of ids [B, T], as one integer: [B, T - n + 1]"""
    T = ids.shape[1]
    k = torch.zeros(ids.shape[0], T - n + 1, dtype=torch.long)
    for j in range(n):
        k = k * V + ids[:, j:T - n + 1 + j]
    return k


def bucket(k, n, rows):
    h = k * MIX[n]                       # (int64 multiply wraps)
    return torch.remainder(h ^ (h >> 29), rows)


class Table:
    """T2, T3 [rows // 2, d], fp32 on the host, with lazy Adam state"""
    def __init__(self, rows, d, V):
        self.T, self.V = {2: torch.zeros(rows // 2, d), 3: torch.zeros(rows // 2, d)}, V
        self.m, self.v, self.t = {}, {}, 0

    def idx(self, ids, n):
        return bucket(keys(ids, n, self.V), n, self.T[n].shape[0])

    def add_to(self, hs, ids, mb=32):
        """hs [B, T, d] += the table's rows for ids [B, T], in place"""
        for b0 in range(0, hs.shape[0], mb):
            for n, T in self.T.items():
                hs[b0:b0 + mb, n - 1:] += T[self.idx(ids[b0:b0 + mb], n)].to(hs.dtype)
        return hs

    def adam(self, g, ids, lr, b1=0.9, b2=0.999, eps=1e-8):
        """one lazy Adam step on the rows ids [B, T] touch, g [B, T, d] the gradient of the stream they enter"""
        self.t += 1
        for n, T in self.T.items():
            u, inv = torch.unique(self.idx(ids, n).reshape(-1), return_inverse=True)
            G = torch.zeros(len(u), T.shape[1]).index_add_(0, inv, g[:, n - 1:].reshape(-1, T.shape[1]).float())
            if n not in self.m:
                self.m[n], self.v[n] = torch.zeros_like(T), torch.zeros_like(T)
            m = self.m[n][u].mul_(b1).add_(G, alpha=1 - b1)
            v = self.v[n][u].mul_(b2).addcmul_(G, G, value=1 - b2)
            self.m[n][u], self.v[n][u] = m, v
            T[u] -= lr * (m / (1 - b1 ** self.t)) / ((v / (1 - b2 ** self.t)).sqrt() + eps)

    def fit(self, t, s, ids, lam, mb=32):
        """closed form: the ridge means of D = t - s (both [B, T, d] on the host), T2 per 2-gram bucket, T3 per 3-gram
        bucket on D - T2[h2]"""
        d = t.shape[-1]
        for n, T in self.T.items():
            S, cnt = torch.zeros_like(T), torch.zeros(T.shape[0])
            for b0 in range(0, t.shape[0], mb):
                i = ids[b0:b0 + mb]
                D = (t[b0:b0 + mb].float() - s[b0:b0 + mb].float())[:, n - 1:]
                if n == 3:
                    D = D - self.T[2][self.idx(i, 2)][:, 1:]
                b = self.idx(i, n).reshape(-1)
                S.index_add_(0, b, D.reshape(-1, d))
                cnt.index_add_(0, b, torch.ones(b.shape[0]))
            T.copy_(S / (cnt[:, None] + lam).clamp_min(1e-6))      # (empty buckets stay 0)
        return self

    def hit(self, train, ids):
        """{n: share of ids' n-gram positions whose bucket a training n-gram touched}"""
        return {n: float(torch.isin(self.idx(ids, n), torch.unique(self.idx(train, n))).float().mean())
                for n in self.T}


def seen(train, ids, V):
    """{n: share of ids' n-gram positions whose n-gram occurs in train}"""
    return {n: float(torch.isin(keys(ids, n, V), torch.unique(keys(train, n, V))).float().mean()) for n in (2, 3)}


def forward(R, layer, hs, ids, tables, dev, mb, lo, hi):
    """hs [B, T, d] (the stream entering layer lo, host) through layer(lo..hi-1), tables {L: Table} added entering L
    (L == hi: added to the output); in place"""
    for i in range(lo, hi + 1):
        if i in tables:
            tables[i].add_to(hs, ids)
        if i < hi:
            stream([layer(i)], hs, dev, mb)
    return hs


def kl_grad(R, h, t, dev, mb):
    """KL(teacher || student) per token at the head, from final streams h (student) and t (teacher) [B, T, d]: the
    mean and its gradient with respect to h (host, fp32)"""
    g, tot = torch.empty(h.shape, dtype=torch.float32), 0.0
    for b0 in range(0, h.shape[0], mb):          # (mb windows of logits [mb * T, vocab] fp32 at once)
        x = h[b0:b0 + mb].to(dev).float().requires_grad_()
        with torch.no_grad():
            lt = F.log_softmax(R.logits(t[b0:b0 + mb].reshape(-1, t.shape[-1])), -1)
        ls = F.log_softmax(R.logits(x.reshape(-1, x.shape[-1])), -1)
        kl = (lt.exp() * (lt - ls)).sum() / (h.shape[0] * h.shape[1])
        kl.backward()
        g[b0:b0 + mb], tot = x.grad.cpu(), tot + float(kl.detach())
    R.__dict__.pop("_head", None)                # (the head's 1 GB off the device during the layer passes)
    return tot, g


def vjp(m, x, g, dev, mb):
    """the gradient of a layer's input x [B, T, d] (host) given its output's g [B, T, d] (host fp32)"""
    m.to(dev)
    out = torch.empty_like(g)
    for b0 in range(0, x.shape[0], mb):
        xi = x[b0:b0 + mb].to(dev).requires_grad_()
        run(m, xi).backward(g[b0:b0 + mb].to(dev, xi.dtype))
        out[b0:b0 + mb] = xi.grad.float().cpu()
    offload(m)
    return out


def train(R, S, tables, x0, tf, ids, a, log):
    """--steps of layer-major distillation of the tables; x0 [N, T, d] the student's stream entering the first
    injection layer, tf [N, T, d] the teacher's final stream, ids [N, T] the training windows"""
    Ls, n, dev = sorted(tables), len(S), a.device
    rng, order, curve = np.random.default_rng(a.seed), [], []
    for step in range(a.steps):
        t0 = time.time()
        if len(order) < a.batch:
            order = list(rng.permutation(len(ids)))
        b, order = torch.tensor(order[:a.batch]), order[a.batch:]
        h, i_b, ins = x0[b].clone(), ids[b], {}
        for i in range(Ls[0], n + 1):
            if i in tables:
                tables[i].add_to(h, i_b)
            if i < n:
                ins[i] = h.clone()
                stream([S[i]], h, dev, a.mb)
        kl, g = kl_grad(R, h, tf[b], dev, 1)
        lr = lr_at(step, a.steps, a.lr, a.lr / 10, a.warmup)
        for i in range(n, Ls[0] - 1, -1):
            if i < n:
                g = vjp(S[i], ins.pop(i), g, dev, a.mb)
            if i in tables:
                tables[i].adam(g, i_b, lr)
        curve.append(kl)
        if step % 10 == 0 or step == a.steps - 1:
            log(f"step {step + 1}: kl {kl:.4f}, lr {lr:.2e}, {time.time() - t0:.1f}s")
    return curve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("hf"); ap.add_argument("conv"); ap.add_argument("data")
    ap.add_argument("--conv-data", required=True, help="the conversion's DATA (its held-out tail is one eval set)")
    ap.add_argument("--cfg", action="append", required=True, help="LAYERS/ROWS[/LAM], e.g. 2+8/2000000")
    ap.add_argument("--tokens", type=int, default=819200, help="training tokens, contiguous windows from the start")
    ap.add_argument("--hold", type=int, default=65536, help="tokens at the end of DATA kept out of training")
    ap.add_argument("--eval-windows", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapters"); ap.add_argument("--act-bits", type=int, default=0)
    ap.add_argument("--student-hf", help="the student's layers from this HF-format directory (decoded weights of "
                    "another quantizer, the teacher's config), folded as the teacher, in place of CONV's")
    ap.add_argument("--mb", type=int, default=2); ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cache", help="directory for the teacher's training streams, reused across runs")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
    R = Run(a.hf, a.conv, a.device)
    cfg, dev, seq = R.cfg, a.device, R.meta["seq"]
    n = a.layers or cfg.num_hidden_layers
    V, d = cfg.vocab_size, cfg.hidden_size
    data = np.memmap(a.data, dtype=np.uint32, mode="r")
    win = lambda lo, k: torch.from_numpy(np.stack([data[s:s + seq].astype(np.int64)
                                                   for s in range(lo, lo + k * seq, seq)]))
    nt = a.tokens // seq
    assert nt * seq <= len(data) - a.hold and a.eval_windows * seq <= a.hold, "training windows reach the held-out tail"
    ids = win(0, nt)
    sets = {"conv": R.held_out(a.conv_data), "tail": win(len(data) - a.eval_windows * seq, a.eval_windows),
            "train": ids[np.linspace(0, nt - 1, a.eval_windows).astype(int)]}
    cov = {k: seen(ids, v, V) for k, v in sets.items()}
    log(f"{nt} training windows ({nt * seq} tokens); seen {cov}")
    ev = torch.cat(list(sets.values()))
    cut = np.cumsum([0] + [len(v) for v in sets.values()])

    def losses(tables):
        hs = forward(R, S.__getitem__, R.embed(ev), ev, tables, dev, a.mb, 0, n)
        return {k: R.loss(hs[cut[j]:cut[j + 1]], ev[cut[j]:cut[j + 1]], a.mb) for j, k in enumerate(sets)}

    rules = None if R.Q is None else C.fold_rules(cfg)
    T = [teacher(R.rd, cfg, i, rules, R.Q, "cpu", R.dt) for i in range(n)]
    if a.student_hf:
        rd = C.Reader(a.student_hf)
        S = [teacher(rd, cfg, i, rules, R.Q, "cpu", R.dt) for i in range(n)]
    else:
        S = [R.layer(i, a.act_bits, a.adapters).requires_grad_(False) for i in range(n)]
    hs = forward(R, T.__getitem__, R.embed(ev), ev, {}, dev, a.mb, 0, n)
    base = {"teacher": {k: R.loss(hs[cut[j]:cut[j + 1]], ev[cut[j]:cut[j + 1]], a.mb) for j, k in enumerate(sets)},
            "student": losses({})}
    log(f"baselines {base}")
    cfgs = []
    for c in a.cfg:
        Ls, rows, lam = (c + "/0").split("/")[:3]
        cfgs.append((sorted(int(x) for x in Ls.split("+")), int(rows), float(lam)))
    need = sorted({L for Ls, _, _ in cfgs for L in Ls} | ({n} if a.steps else set()))
    path = lambda L: os.path.join(a.cache, f"teacher_L{L}_{nt}x{seq}.pt") if a.cache else ""
    snap = {L: torch.load(path(L)) for L in need if os.path.exists(path(L))}    # the teacher's stream entering each
    miss = [L for L in need if L not in snap]                                    # injection layer (training: its final)
    if miss:
        hs = R.embed(ids)
        for i in range(miss[-1] + 1):
            if i in miss:
                snap[i] = hs.clone()
                if a.cache:
                    torch.save(snap[i], path(i))
            if i < miss[-1]:
                stream([T[i]], hs, dev, a.mb)
        del hs
    log(f"teacher streams at {need} ({len(need) - len(miss)} from the cache)")
    results = []
    for Ls, rows, lam in cfgs:
        t0, tables = time.time(), {L: Table(rows, d, V) for L in Ls}
        x0 = forward(R, S.__getitem__, R.embed(ids), ids, {}, dev, a.mb, 0, Ls[0])
        if a.steps:
            curve = train(R, S, tables, x0, snap[n], ids, a, log)
        else:
            curve = None
            for j, L in enumerate(Ls):
                tables[L].fit(snap[L], x0, ids, lam).add_to(x0, ids)
                if j + 1 < len(Ls):
                    forward(R, S.__getitem__, x0, ids, {}, dev, a.mb, L, Ls[j + 1])
        del x0
        r = {"layers": Ls, "rows": rows, "lam": lam if not a.steps else None, "steps": a.steps, "batch": a.batch,
             "lr": a.lr, "params": len(Ls) * (rows // 2) * 2 * d, "tokens": nt * seq,
             "loss": losses(tables), "baseline": base, "seen": cov,
             "hit": {k: tables[Ls[0]].hit(ids, v) for k, v in sets.items()},
             "kl_first10": float(np.mean(curve[:10])) if curve else None,
             "kl_last10": float(np.mean(curve[-10:])) if curve else None,
             "adapters": a.adapters, "student_hf": a.student_hf, "act_bits": a.act_bits, "seconds": round(time.time() - t0)}
        print(json.dumps(r), flush=True)
        results.append(r)
    return results


if __name__ == "__main__":
    main()
