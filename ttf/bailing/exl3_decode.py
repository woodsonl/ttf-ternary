"""Decode an ExLlamaV3 (EXL3) Ling-3.0 conversion to dense HF-format decoder layers, for ttf.bailing.ngram --student-hf.

Runs in ExLlamaV3's own environment (its CUDA extension reconstructs the trellis-coded weights). Every quantized linear
(KEY.trellis with KEY.suh / KEY.svh / KEY.mul1) becomes KEY.weight [out, in] through LinearEXL3.get_weight_tensor; the
fused attention qkv_proj and conv1d split back into the Hub's q / k / v tensors by the Hub's shapes; everything else is
copied. Output: OUT/model-L{i}.safetensors per decoder layer and model.safetensors.index.json, in the Hub's names and
dtypes (the layers only: embedding and head stay the teacher's). Prints each layer's relative weight error against
the Hub's float weights.
  python exl3_decode.py HF_DIR EXL3_DIR OUT_DIR
"""
import glob, json, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav3.modules.quant.exl3 import LinearEXL3

SPLIT = {"attention.qkv_proj": ("attention.q_proj.weight", "attention.k_proj.weight", "attention.v_proj.weight"),
         "attention.conv1d.weight": ("attention.q_conv1d.weight", "attention.k_conv1d.weight",
                                     "attention.v_conv1d.weight")}


def decode(hf, exl, out, dev="cuda"):
    hub = json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]
    src = {}
    for f in glob.glob(os.path.join(exl, "*.safetensors")):
        with safe_open(f, "pt") as g:
            src.update({k: f for k in g.keys()})
    get = lambda k: safe_open(src[k], "pt").get_tensor(k)
    ref = lambda k: safe_open(os.path.join(hf, hub[k]), "pt").get_tensor(k)
    os.makedirs(out, exist_ok=True)
    wmap, n = {}, 1 + max(int(k.split(".")[2]) for k in hub if k.startswith("model.layers."))
    for li in range(n):
        pre, sd = f"model.layers.{li}.", {}
        for k in sorted(x for x in src if x.startswith(pre)):
            if k.endswith(".trellis"):
                key = k[:-len(".trellis")]
                t = {s: get(f"{key}.{s}").to(dev) for s in ("trellis", "suh", "svh", "mul1", "mcg") if f"{key}.{s}" in src}
                lin = LinearEXL3(None, t["suh"].numel(), t["svh"].numel(), key=key, **t)
                sd[key + ".weight"] = lin.get_weight_tensor().T.float().cpu()
            elif not k.endswith((".suh", ".svh", ".mul1", ".mcg")):
                sd[k] = get(k)
        for f, parts in SPLIT.items():
            k = next((x for x in list(sd) if x == pre + f or x == pre + f + ".weight"), None)
            if k is None:
                continue
            sizes = [ref(pre + p).shape[0] for p in parts]
            for p, v in zip(parts, sd.pop(k).split(sizes, 0)):
                sd[pre + p] = v.reshape(ref(pre + p).shape)
        want = {k for k in hub if k.startswith(pre)}
        assert set(sd) == want, (li, sorted(set(sd) ^ want)[:8])
        err, num = 0.0, 0.0
        for k in sd:
            r = ref(k)
            sd[k] = sd[k].to(r.dtype).contiguous()
            if k.endswith("proj.weight"):
                err, num = err + float((sd[k].float() - r.float()).pow(2).sum()), num + float(r.float().pow(2).sum())
        f = f"model-L{li}.safetensors"
        save_file(sd, os.path.join(out, f), metadata={"format": "pt"})
        wmap.update({k: f for k in sd})
        print(f"layer {li}: {len(sd)} tensors, linear rel err {(err / num) ** 0.5:.4f}", flush=True)
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    for f in ("config.json", "configuration_bailing_moe_v3.py"):
        if os.path.exists(os.path.join(hf, f)):
            open(os.path.join(out, f), "w").write(open(os.path.join(hf, f)).read())


if __name__ == "__main__":
    with torch.no_grad():
        decode(*sys.argv[1:4])
