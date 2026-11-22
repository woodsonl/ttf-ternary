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
- `ngram.py`: the n-gram recovery table (below). Every weight frozen; hashed 2-gram and 3-gram rows added to the
  residual stream entering chosen layers, trained by KL distillation from the float teacher with lazy host Adam on the
  touched rows; `--steps 0` is the closed-form ridge ablation; `--cache` keeps the teacher's training streams.
- `exl3_decode.py`: an ExLlamaV3 (EXL3) conversion decoded to dense HF-format layers for `ngram.py --student-hf`; runs
  in ExLlamaV3's environment and trims its zero-row output padding (`kv_a_proj_with_mqa` 640 to 576).
- `tests/test_bailing.py`: a tiny random bailing_hybrid converted on CPU; Stage 1 parity, one adapter-training run, and `test_ngram_table`.

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
- `train.py` keeps fp32 adapters and fp32 AdamW moments (12 bytes a parameter before gradients): ~30 GB of host memory
  for flash's 2.53 B adapter parameters, above the lean-optimizer figure in the memory section.
- KDA runs transformers' reference torch path (`flash-linear-attention` is absent); speed against `fla` is unmeasured.
- Training ran at act 0; act 8 costs 0.005 nats after training. Training at act 8 is untried.
- GPU bf16 losses jitter in the fourth decimal (CUDA `index_add` is nondeterministic).

### N-gram recovery table on time3

`python -m ttf.bailing.ngram HF T3 WIKI --conv-data DATA --tokens 819200 --cache CACHE --steps 200 --lr 1e-2 --cfg ...`
(add `--adapters DIR` or `--student-hf DIR`). Training: the first 819,200 tokens of `wiki.bin` (1,600 windows of 512),
200 steps of batch 8, one epoch. Losses are nats; three eval sets: conv (the conversion's 4 held-out windows of
`local_text.bin`), tail (16 windows from the end of `wiki.bin`, held out of training), train (16 training windows).
"Closed" is the share of the student-to-teacher gap the row recovers.

| student | recovery | params | conv | tail | train | closed conv / tail / train |
|---|---|---|---|---|---|---|
| float teacher | none | | 2.084 | 2.269 | 2.452 | |
| ternary (time3) | none | | 3.599 | 5.225 | 5.162 | |
| ternary | wiki adapters, rank 16 | 227.3 M | 3.795 | 3.728 | 3.825 | -13% / 51% / 49% |
| ternary | table, layer 2, 2 M rows | 3.07 B | 3.402 | 3.460 | 3.118 | 13% / 60% / 75% |
| ternary | table, layer 2, 0.5 M rows | 0.77 B | 3.530 | 3.553 | 3.184 | 5% / 57% / 73% |
| ternary | table, layers 2+8+14+20, 0.5 M rows each | 3.07 B | 3.530 | 3.456 | 3.050 | 5% / 60% / 78% |
| ternary | table, layer 20, 2 M rows | 3.07 B | 3.522 | 4.410 | 4.260 | 5% / 28% / 33% |
| ternary | closed form, layer 2, 2 M rows, lam 1 | 3.07 B | 3.649 | 5.107 | 4.876 | -3% / 4% / 11% |
| ternary | closed form, layer 20, 2 M rows, lam 1 | 3.07 B | 3.684 | 4.544 | 3.081 | -6% / 23% / 77% |
| ternary | wiki adapters + table 2+8+14+20 | 3.30 B | 3.617 | 3.215 | 2.933 | -1% / 68% / 82% |
| EXL3 1.66 bpw | none | | 2.479 | 2.652 | 2.814 | |
| EXL3 1.66 bpw | table 2+8+14+20, lr 1e-2 | 3.07 B | 2.601 | 2.509 | 2.752 | -31% / 37% / 17% |
| EXL3 1.66 bpw | table 2+8+14+20, lr 1e-3 | 3.07 B | 2.405 | 2.418 | 2.483 | 19% / 61% / 91% |

Every trained row used 819,200 tokens. The adapters: `python -m ttf.bailing.train HF T3 WIKI OUT --steps 200
--eval-every 50 --mb 2 --hold 3180854` (wiki.bin's 4,000,054 tokens less 819,200), 200 steps of 8 random windows of
512 from the same 819,200-token prefix. Closed form: `--steps 0 --cfg 2/2000000/1 --cfg 20/2000000/1`.
Coverage, the share of eval positions whose n-gram occurs in training ("seen") and whose bucket a training n-gram
touched ("hit", collisions included), 2-gram / 3-gram:

| set | seen | hit, 2 M rows | hit, 0.5 M rows |
|---|---|---|---|
| conv | 33% / 13% | 52% / 51% | 79% / 90% |
| tail | 65% / 36% | 75% / 62% | 91% / 91% |

Mean KL, first 10 / last 10 steps: layer 2 2 M 2.73 / 1.46; 2+8+14+20 2.58 / 1.47; layer 20 3.15 / 2.31; with adapters
1.59 / 1.26; EXL3 lr 1e-2 0.60 / 0.65; EXL3 lr 1e-3 0.63 / 0.48. A step is 26 to 31 s on GPU 1; a run is 1.5 to 1.7 h
plus a 13 min teacher pass the cache saves.

The EXL3 student comes from ExLlamaV3 v1.6.7, `convert.py -b 1.66 -hb 16 -eb 16` (head and embeddings 16-bit), decoded
by `exl3_decode.py` (linear relative weight error 0.24 to 0.39 per layer).

What the table learns: it generalizes inside its training domain and barely outside it. On held-out wiki it recovers
about 60% of the gap on both students, against 78% (ternary) and 91% (EXL3) on its own training windows; on conv, whose
3-grams are 13% seen, 5 to 19%. The closed form at layer 20 is the memorization control: it fits the training windows
(77%) as well as gradient training does and recovers 23% of held-out wiki and none of conv. Early injection beats late
(layer 20 trained: 28% of tail). Fewer rows raise hit rates but add collisions and cost conv (0.5 M against 2 M at
layer 2). On the ternary student the wiki-trained adapters cost conv 0.20 nats and the table does not win it back; the
conv-trained adapters of the section above (2.924, 44% of the conv gap) remain the better conv recovery.

Unverified for the table:

- lr 1e-2 was chosen for the ternary student and never swept there; on EXL3 lr 1e-3 beat it on every set, so the
  ternary rows may also be under-tuned.
- One seed, one token budget (819,200), one epoch; collisions between rows are not separated from generalization.
- The 2+8+14+20 configuration was picked by lowest tail loss and lr 1e-3 after seeing held-out results; conv is the
  only set nothing was selected on.
- The recipes differ beyond the token count: the adapters train layerwise on `|s - t|^2 / |t - h|^2` at lr 1e-3 to
  1e-4 cosine, the table end to end on KL at lr 1e-2 (or 1e-3) with warmup and cosine decay.
- The decoded EXL3 student is checked only by per-layer weight error, not against ExLlamaV3's own inference loss.
- The adapters and the table see the same tokens but differ 13x in parameters; a parameter-matched comparison is unrun.
- Ling-3.0-flash and Sante are unrun; table memory (3.07 B fp32 rows plus Adam state, on the host) scales with rows,
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
