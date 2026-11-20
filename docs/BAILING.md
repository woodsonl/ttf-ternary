# ttf.bailing: Ling-3.0-flash in TTF

Ling-3.0-flash (HF `model_type` bailing_hybrid): 42 layers, KDA everywhere except each sixth layer, which is gated
MLA; 512 routed experts + 1 shared per MoE layer, sigmoid `noaux_tc` router in fp32, untied head. 127.5 B parameters,
5.1 B active. The converter side lives in `ttf-conversion` (`ttfconv/arch.py` discovery, `docs/BAILING.md` there).

## What the architecture changes about the port

Two of the three unknowns this port was expected to carry are gone, both because transformers 5.23 ships the family
natively (`BailingHybridConfig` / `BailingHybridForCausalLM`, pure torch, no `fla`):

- **The attention backward.** `BailingHybridDeltaAttention` is ordinary autograd-able torch, so a layer-major trainer
  does not have to borrow `fla`'s chunk kernels and does not have to find or write a KDA backward. A full-sequence
  forward takes the chunked form (`chunk_kimi_delta_attention`); its speed against `fla`'s Triton kernels is not
  measured yet.
- **The fold.** `ttfconv.arch` discovers the residual rotation as exact for this family (`fold_err` 3.3e-06, gain form
  `w`), so the trainer sees a folded model: norm gains already in the readers, one shared Hadamard on the residual
  stream, no per-branch rotation at decode.

What is still open is the trainer proper: the layer-major loop over 42 layers with the model and the TTF state in host
memory, the fused-expert grouped GEMMs, and the KD head.

## Memory: the reason this is not a `q38` copy

`ttf.q38.train` holds ~221 GB pinned on a 96 GB card for Qwen3.8-Flash-Next, at a TTF rank of ~640. The TTF accumulator
is `A = U V^T` per linear, so state is `2 * rank * sum(N + K)` bytes over the converted matrices. Ling spreads 126.6 B
parameters over **42,359 matrices** (most of them expert `[768, 2560]` and `[2560, 768]`), which is the worst ratio of
state to parameters of any model in this repo's history:

| rank | TTF state (bf16 U, V) |
|---|---|
| 128 | 40 GB |
| 256 | 81 GB |
| 512 | 162 GB |
| 640 | 202 GB |

Computed from `sum(N + K) = 157.9 M` over the 42,359 converted matrices, not measured. The cross-check that makes the
formula credible: at rank 640 it reproduces the measured ~221 GB for Qwen3.8-Flash-Next within 1%.

The consequence for a single workstation: a rank-640 run wants ~200 GB of pinned host memory, matching the q38 design,
while a rank-256 run fits in 81 GB. The shipped 27B release avoided the question entirely by freezing the trits and
training only the rank-16 float adapters (`adapter_ctl:mode=frozen,main=frozen`), which needs no TTF state at all.
That is the first target here too; the float adapters are 2.53 B parameters (`16 * sum(N + K)`), ~6.6 GB with a lean
optimizer.

## Token budget

No number here is inherited from the 27B recipe, because the 27B's 2,200 steps fitted float adapters on frozen trits
and therefore say nothing about TTF convergence. The two measured recovery runs, as a crude normalisation by active
parameters per token:

| run | params/token | tokens | tokens per active param |
|---|---|---|---|
| Qwen3.5-0.8B | 0.8 B dense | 32.8 M | 0.041 |
| Qwen3.8-27B, released | 27 B dense | 72 M | 0.0027 |
| Ling-3.0-flash | 5.1 B active | ? | — |

Set the budget from a 200-step held-out loss curve rather than from either row. Two factors push in opposite directions:
the conversion damage is concentrated in 40 MoE layers whose routing shifts as trits move; against that, each expert
sees 1.56% of tokens, so 72 M tokens already gives ~1.1 M tokens per expert.

## Recipe

Target: `recipes/bailing-3.0-flash.toml`, descending from `recipes/continue-27b.toml` rather than from the q38 recipe,
since the first target is the frozen-trit float-adapter run. Folded input, `--ttf-offload async`, a slim save format
(the q38 full checkpoint is 139 GB + optimizer and will not fit beside a 55 GB KD cache).
