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

## What exists

`ttf/bailing/` runs a `ttfconv.bailing.convert` output (`OUT/L{i}.pt`) layer-major, one decoder layer on the device at
a time, everything else on the host:

- `student.py`: one converted layer as a module. transformers' one-layer shell, `TernLinear` for each dense matrix,
  two `TernaryExperts` (gate_up, down) for the routed experts run sorted by expert, the rest from `_float`, all loaded
  strictly. `teacher()` is the Hub layer folded as the converter folds it. `lora=r` adds a rank-r float adapter
  (A at zero) beside every ternary matrix.
- `eval.py`: the conversion's held-out windows through every layer, then `convert.eval_loss`'s head; `--prompt` does
  greedy generation. `--act-bits 0` turns off activation quantisation (the converter's own fake-quantised layer, the
  number in its `eval.json`); 8 is per-token int8, as the TTF kernels run. CUDA runs bf16, CPU fp32.
- `train.py`: frozen-trit recovery. Teacher and student read the same input (the student's stream); the student's
  adapters take one AdamW step per layer per step on `|s - t|^2 / |t - h|^2`; optimizer state lives on the host.
- `ttf/q38/moe.py`: `act_bits=0` in `TernaryExperts` (padded path only).
- `tests/test_bailing.py`: a tiny random bailing_hybrid converted on CPU; Stage 1 parity and one adapter-training run.

### Measured on time3

time3 is a conversion of Ling-3.0-tiny (24 layers, 128 experts, hidden 1536; rank 16, seed 777, 4 held-out windows
of 512 tokens). Workstation, GPU 1 (RTX 4060 Ti 8 GB) or CPU, `PYTHONPATH` set to both repos.

| what | value | command |
|---|---|---|
| converter `eval.json` ternary | 3.595355069 | (from the conversion) |
| converter `eval.json` float | 2.083564 | (from the conversion) |
| student, CPU fp32, act 0 | 3.595354830 (diff -2.4e-7; final stream rel err 6.6e-7) | `python -m ttf.bailing.eval HF T3 DATA --act-bits 0 --device cpu` |
| student, CPU fp32, act 8 | 3.602300 | same, `--act-bits 8` |
| student, GPU bf16, act 0 / act 8 | 3.602889 / 3.605475 | same, `--device cuda` |
| 200 adapter steps, GPU held-out | 2.922553 | `python -m ttf.bailing.train HF T3 DATA OUT --steps 200 --eval-every 20 --mb 2` |
| trained adapters, CPU fp32, act 0 | 2.923788 | `python -m ttf.bailing.eval HF T3 DATA --act-bits 0 --adapters OUT --device cpu` |
| trained adapters, CPU fp32, act 8 | 2.928834 | same, `--act-bits 8` |

The 200 steps (batch 8 x 512 = 819 K tokens, rank 16, 227.3 M adapter parameters, lr 1e-3 to 1e-4 cosine, 33 s a
step) recover 44% of the ternary-to-float gap (3.5954 to 2.9238 against a float 2.0836). GPU held-out loss by step:

| step | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out | 3.6035 | 3.2732 | 3.1637 | 3.1100 | 3.0510 | 3.0192 | 2.9559 | 2.9715 | 2.9450 | 2.9320 | 2.9226 |

Mean layer KD loss went 0.2455 (step 1) to 0.1570 (step 200). The curve is still falling at step 200, so the token
budget stays open. Greedy generation: the folded float teacher through the same path gives "The capital of France is
Paris."; the untrained ternary student degenerates.

### Unverified

- All numbers above are Ling-3.0-tiny; Ling-3.0-flash and Sante are unrun.
- `train.py` holds every folded teacher in host memory, which a 127.5 B model exceeds; the flash run needs teachers
  folded per layer on the fly or a disk cache.
- KDA runs transformers' reference torch path (`flash-linear-attention` is absent); speed against `fla` is unmeasured.
- Training ran at act 0; act 8 costs 0.005 nats after training. Training at act 8 is untried.
- GPU bf16 losses jitter in the fourth decimal (CUDA `index_add` is nondeterministic).

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
