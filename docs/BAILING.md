# ttf.bailing: Ling-3.0-flash in TTF

Ling-3.0-flash (HF `model_type` bailing_hybrid): 42 layers, KDA everywhere except each sixth layer, which is gated
MLA; 512 routed experts + 1 shared per MoE layer, sigmoid `noaux_tc` router in fp32, untied head. 127.5 B parameters,
5.1 B active. The converter side lives in `ttf-conversion` (`ttfconv/arch.py` discovery, `docs/BAILING.md` there).

## What the architecture changes about the port

Two of the three unknowns this port was expected to carry are gone, both because transformers 5.23 ships the family
natively (`BailingHybridConfig` / `BailingHybridForCausalLM`):

- **The attention backward.** `BailingHybridDeltaAttention` is ordinary autograd-able torch, and with
  `flash-linear-attention` installed its chunked form calls `fla`'s `chunk_kda`, so a layer-major trainer does not have
  to write a KDA backward. Every number below ran with `fla` 0.6.0 on CUDA.
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
  number in its `eval.json`); 8 is per-token int8, as the TTF kernels run.
- `train.py`: frozen-trit recovery. Teacher and student read the same input (the student's stream); the student's
  adapters take one AdamW step per layer per step on `|s - t|^2 / |t - h|^2`; optimizer state lives on the host.
- `ttf/q38/moe.py`: `act_bits=0` in `TernaryExperts` (padded path only).
- `ngram.py`: the n-gram recovery table (below). Every weight frozen; hashed 2-gram and 3-gram rows added to the
  residual stream entering chosen layers, trained by KL distillation from the float teacher with lazy host Adam on the
  touched rows; `--steps 0` is the closed-form ridge ablation; `--cache` keeps the teacher's training streams (the
  directory must exist).
- `exl3_decode.py`: an ExLlamaV3 (EXL3) conversion decoded to dense HF-format layers for `ngram.py --student-hf`; runs
  in ExLlamaV3's environment and trims its zero-row output padding (`kv_a_proj_with_mqa` 640 to 576).
- `tests/test_bailing.py`: a tiny random bailing_hybrid converted on CPU; Stage 1 parity, one adapter-training run, and `test_ngram_table`.

### Measured on time3

time3 is a conversion of Ling-3.0-tiny (24 layers, 128 experts, hidden 1536; rank 16, seed 777, 16 calibration
windows and 4 held-out windows of 512 tokens of `local_text.bin`): `python -m ttfconv.bailing.convert HF DATA T3
--windows 16 --seq 512 --eval-windows 4 --device cuda`, 4.5 min. Every number in this file was measured on one
workstation: RTX 5080 16 GB, 123 GB host memory, `PYTHONPATH` set to both repos.

| what | value | command |
|---|---|---|
| converter `eval.json` float | 2.0851 | (from the conversion) |
| converter `eval.json` ternary | 3.5519 | (from the conversion) |
| 200 adapter steps, GPU bf16 held-out | 3.5622 at step 0 to 2.9601 | `python -m ttf.bailing.train HF T3 DATA OUT --steps 200 --eval-every 20 --mb 2` |

The 200 steps (batch 8 x 512 = 819 K tokens, rank 16, 227.3 M adapter parameters, lr 1e-3 to 1e-4 cosine, 13.6 s a
step) recover 41% of the ternary-to-float gap (3.5622 to 2.9601 against a float 2.0851). GPU held-out loss by step:

| step | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out | 3.5622 | 3.2634 | 3.1398 | 3.0924 | 3.0673 | 3.0483 | 2.9835 | 2.9920 | 2.9737 | 2.9604 | 2.9601 |

Mean layer KD loss went 0.2424 (step 1) to 0.1572 (step 200). The held-out loss is flat over the last 20 steps.

### Unverified

- Ling-3.0-flash and Sante have conversions (`ttf-conversion`, `docs/BAILING.md`) but no recovery run.
- `train.py` holds every folded teacher in host memory, which a 127.5 B model exceeds; the flash run needs teachers
  folded per layer on the fly or a disk cache.
- `train.py` keeps fp32 adapters and fp32 AdamW moments (12 bytes a parameter before gradients): ~30 GB of host memory
  for flash's 2.53 B adapter parameters, above the lean-optimizer figure in the memory section.
- `eval.py --device cpu` fails with `fla` installed (Triton finds no driver), so the CPU fp32 parity check needs
  `arch.cpu_reference_kernels` before the student is built.
- Training ran at act 0; training at act 8 is untried.
- GPU bf16 losses jitter in the fourth decimal (CUDA `index_add` is nondeterministic).

### N-gram recovery table on time3

`python -m ttf.bailing.ngram HF T3 WIKI --conv-data DATA --tokens 819200 --cache CACHE --steps 200 --lr LR
--cfg 2+8+14+20/500000` (add `--student-hf DIR` for the EXL3 student). Training: the first 819,200 tokens of
`wiki.bin` (1,600 windows of 512), 200 steps of batch 8, one epoch. Losses are nats; three eval sets: conv (the
conversion's 4 held-out windows of `local_text.bin`), tail (16 windows from the end of `wiki.bin`, held out of
training), train (16 training windows). "Closed" is the share of the student-to-teacher gap the row recovers. Every
table is layers 2+8+14+20 with 0.5 M rows each, 3.07 B parameters.

| student | recovery | conv | tail | train | closed conv / tail / train |
|---|---|---|---|---|---|
| float teacher | none | 2.085 | 2.270 | 2.451 | |
| ternary (time3) | none | 3.554 to 3.559 | 5.239 | 5.237 | |
| ternary | table, lr 1e-2 | 3.551 | 3.468 | 2.998 | 1% / 60% / 80% |
| ternary | table, lr 1e-3 | 3.287 | 3.440 | 3.202 | 18% / 61% / 73% |
| EXL3 1.66 bpw | none | 2.487 | 2.676 | 2.816 | |
| EXL3 1.66 bpw | table, lr 1e-2 | 2.592 | 2.506 | 2.765 | -26% / 42% / 14% |
| EXL3 1.66 bpw | table, lr 1e-3 | 2.402 | 2.399 | 2.471 | 21% / 68% / 94% |

Coverage, the share of eval positions whose n-gram occurs in training ("seen") and whose bucket a training n-gram
touched ("hit", collisions included), 2-gram / 3-gram:

| set | seen | hit |
|---|---|---|
| conv | 33% / 13% | 79% / 90% |
| tail | 65% / 36% | 91% / 91% |

Mean KL, first 10 / last 10 steps: ternary 2.62 / 1.46 (lr 1e-2), 3.09 / 1.44 (lr 1e-3); EXL3 0.60 / 0.66 (lr 1e-2),
0.63 / 0.47 (lr 1e-3). A step is 9.2 s for the ternary student; a run is 32 min (ternary) or 50 min (EXL3), plus a
4 min teacher pass the cache saves.

The EXL3 student comes from ExLlamaV3 v1.6.7, `convert.py -b 1.66 -hb 16 -eb 16` (head and embeddings 16-bit), decoded
by `exl3_decode.py` (linear relative weight error 0.24 to 0.40 per layer).

What the table learns: it generalizes inside its training domain and barely outside it. On held-out wiki it recovers
60% to 68% of the gap, against 73% to 94% on its own training windows; on conv, whose 3-grams are 13% seen, -26% to
21%. At lr 1e-2 the EXL3 student loses conv (2.487 to 2.592); lr 1e-3 is the better rate on both students for conv and
tail.

Unverified for the table:

- One seed, one token budget (819,200), one epoch, two learning rates; collisions between rows are not separated from
  generalization.
- The 2+8+14+20 configuration with 0.5 M rows was picked on held-out results; conv is the only set nothing was selected
  on.
- The recipes differ beyond the token count: the adapters train layerwise on `|s - t|^2 / |t - h|^2` at lr 1e-3 to
  1e-4 cosine, the table end to end on KL with warmup and cosine decay.
- The decoded EXL3 student is checked only by per-layer weight error, not against ExLlamaV3's own inference loss.
- The table has not run on Ling-3.0-flash or Sante; its memory (3.07 B fp32 rows plus Adam state, on the host) scales with rows,
  not with the model.

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
