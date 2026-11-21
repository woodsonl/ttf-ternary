"""Frozen-trit recovery of a converted Ling-3.0 run (ttfconv.bailing.convert's OUT): layer-major distillation of
float adapters.

One step: --batch training windows (random, from DATA without the conversion's held-out tail) enter as emb @ Q^T; at
each layer the float teacher (the Hub's layer, folded as the conversion folded it: ttf.bailing.student.teacher) and
the ternary student (ttf.bailing.student.student: trits, scales and the ternary adapters frozen, a rank --rank float
adapter A B^T beside every ternary matrix, A at 0) read the same input, the student's stream. The student's adapters
take one AdamW step on the layer loss |s - t|^2 / |t - h|^2 (the student's output against the teacher's, relative to
the teacher layer's own update), and the student's output is the next layer's input. One layer's teacher and student
on the device at a time; all teachers, students and optimizer states on the host.

Every --eval-every steps: the held-out loss through the students (ttf.bailing.eval: the conversion's held-out windows
and head). At step 0 the adapters are zero, so it is ttf.bailing.eval's number for the run; the teachers' held-out
loss is logged once (the conversion's eval.json "float").
Output: OUT/A{i}.pt (the adapters, for ttf.bailing.eval --adapters), OUT/curve.json.
  python -m ttf.bailing.train HF_DIR CONV DATA.bin OUT [--steps 200] [--batch 8] [--lr 1e-3] [--rank 16]
"""
import argparse, json, os, time
import numpy as np
import torch

from ttf.bailing.eval import Run, stream
from ttf.bailing.student import student, teacher, adapters, offload, run
from ttf.q38.train import lr_at
from ttfconv.bailing import convert as C


def opt_to(opt, dev):
    """an optimizer's moment tensors to dev (the step counts stay on the host)"""
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v) and v.dim():
                st[k] = v.to(dev)


def layer_step(T, S, opt, h, dev, mb, lr):
    """one layer: teacher and student on the micro-batches of h [B, seq, d] (host), the student's adapters one AdamW
    step; returns (the student's outputs on the host, the mean layer loss)"""
    T.to(dev); S.to(dev); opt_to(opt, dev)
    for g in opt.param_groups:
        g["lr"] = lr
    out, tot, n = torch.empty_like(h), 0.0, -(-h.shape[0] // mb)
    for b0 in range(0, h.shape[0], mb):
        x = h[b0:b0 + mb].to(dev)
        with torch.no_grad():
            t = run(T, x).float()
            den = (t - x.float()).pow(2).mean()
        s = run(S, x)
        loss = (s.float() - t).pow(2).mean() / den
        (loss / n).backward()
        tot += float(loss.detach()) / n
        out[b0:b0 + mb] = s.detach().to("cpu", h.dtype)
    opt.step(); opt.zero_grad(set_to_none=True)
    offload(T); offload(S); opt_to(opt, "cpu")
    return out, tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("hf"); ap.add_argument("conv"); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=0, help="training window (0: the conversion's)")
    ap.add_argument("--mb", type=int, default=4); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--act-bits", type=int, default=0, help="0: as the conversion's eval.json; 8: int8 activations")
    ap.add_argument("--eval-every", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--hold", type=int, default=0, help="tokens at the end of DATA kept out of training (at least the "
                    "conversion's held-out windows)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
    os.makedirs(a.out, exist_ok=True)
    R = Run(a.hf, a.conv, a.device)
    cfg, dev = R.cfg, a.device
    n = a.layers or cfg.num_hidden_layers
    rules = None if R.Q is None else C.fold_rules(cfg)
    T = [teacher(R.rd, cfg, i, rules, R.Q, "cpu", R.dt) for i in range(n)]
    S = []
    for i in range(n):
        blob = torch.load(os.path.join(a.conv, f"L{i}.pt"), map_location="cpu", weights_only=True)
        S.append(student(cfg, i, blob, R.dt, lora=a.rank, act_bits=a.act_bits))
    opts = [torch.optim.AdamW(list(adapters(m).values()), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd) for m in S]
    log(f"{n} layers: {sum(p.numel() for m in S for p in adapters(m).values()) / 1e6:.1f} M adapter parameters "
        f"(rank {a.rank}), act bits {a.act_bits}, {dev} {R.dt}")
    ids_e = R.held_out(a.data)
    held_loss = lambda L: R.loss(stream(L, R.embed(ids_e), dev, a.mb), ids_e, a.mb)
    log(f"teacher held-out loss {held_loss(T):.6f}")
    data = np.memmap(a.data, dtype=np.uint32, mode="r")
    seq = a.seq or R.meta["seq"]
    hi = len(data) - max(a.hold, R.meta["eval_windows"] * R.meta["seq"]) - seq - 1     # (no window reaches the tail)
    rng = np.random.default_rng(a.seed)
    curve = []
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            ev = held_loss(S)
            curve.append({"step": step, "held_out": ev})
            log(f"step {step}: held-out loss {ev:.6f}")
            json.dump(curve, open(os.path.join(a.out, "curve.json"), "w"), indent=1)
        if step == a.steps:
            break
        t0 = time.time()
        lr = lr_at(step, a.steps, a.lr, a.min_lr, a.warmup)
        ids = torch.from_numpy(np.stack([data[s:s + seq].astype(np.int64) for s in rng.integers(0, hi, a.batch)]))
        h, kd = R.embed(ids), []
        for i in range(n):
            h, l = layer_step(T[i], S[i], opts[i], h, dev, a.mb, lr)
            kd.append(l)
        curve.append({"step": step + 1, "kd": float(np.mean(kd)), "kd_layers": kd, "lr": lr})
        log(f"step {step + 1}: kd {np.mean(kd):.4f} (first {kd[0]:.4f}, last {kd[-1]:.4f}), lr {lr:.2e}, "
            f"{time.time() - t0:.1f}s")
    for i, m in enumerate(S):
        torch.save({k: v.detach().cpu() for k, v in adapters(m).items()}, os.path.join(a.out, f"A{i}.pt"))
    log(f"adapters saved to {a.out}")
    return curve


if __name__ == "__main__":
    main()
