# one-way-door

A weekend reproduction, in miniature, of Anthropic's *Teaching Claude why*
finding: training a model on the **reasoning behind** a behavior generalizes
out-of-distribution better than training on the behavior alone.

We teach one concrete principle — **favor reversible actions; when a choice is a
one-way door, slow down** (you can always add more salt, you can't take it out) —
and ask whether a small model carries it into domains it never trained on.

Full write-up: *(blog post forthcoming)*.

## How it works

- **Principle.** Favor reversible actions. When a move is hard to undo and a
  reversible alternative exists, do the reversible thing first (taste before you
  salt; trim less before you cut more; sleep on the angry email). Crucially the
  correct call is *not* always "wait" — so the task is a **2-class decision**,
  `reversible_first` vs `commit_now`, scored on the action.
- **Taxonomy (two-tier).** Exactly one axis decides the answer; the rest are
  balanced decoys. **Tier 1 — situation structure** sets the label: a true
  one-way-door-with-escape → `reversible_first`; *delay is the one-way door* or a
  *false door* (feels permanent, is recoverable) → `commit_now`. **Tier 2 —
  decoys** (irreversibility type · trap · perspective · domain) are sampled
  independently of the structure and **balanced within each label**, so no surface
  cue predicts the answer — the only way to score is to read the structure, which
  is the principle. (`reversible_move` is recorded, not balanced.) See
  `data/taxonomy.json`; the realised balance is written to
  `results/dataset/balance.json`.
- **Data.** Claude synthesizes a dilemma per scenario (in the scenario's
  perspective) and one recommended action rendered three ways: **demos** (the
  action alone), **placebo** (the same action padded to the length of `why` with
  reasoning-free filler), and **why** (the action plus the reversibility
  reasoning). The placebo is a length-matched control — see below.
- **Reasoning order.** By default (`WHY_ORDER="reason_first"`) the `why` target
  puts the reasoning *before* the action, so it is causally upstream — the model
  can re-derive the answer and training conditions the action on the reasoning.
  `placebo` is filler-first to match, keeping the action in the same position.
  Flip to `"action_first"` for the stricter internalization ablation (the action
  is emitted before any reasoning, so a gap can't be test-time chain-of-thought).
- **Control.** An independent Claude pass verifies all three responses recommend
  the *same* action, that `demos` and `placebo` carry **no** reversibility
  reasoning while `why` does, and re-derives the answer key.
- **Train.** One small base model, LoRA-fine-tuned once per arm (demos / placebo /
  why) with identical seeds, hyper-parameters, and prompts. The native `<think>`
  scratchpad is disabled so any difference comes from the training targets.
- **Decompose.** With three arms the held-out gap splits cleanly:
  `placebo − demos` is the effect of *longer output*, `why − placebo` is the
  effect of the *reasoning content*. If only the second clears zero, the transfer
  is the reasoning, not the extra tokens — the confound the two-arm design can't
  rule out. `control_report.json` reports the demos/placebo/why word counts so you
  can confirm the placebo is actually length-matched.
- **Probe.** Both adapters (and the untrained base) answer eval dilemmas with
  **no mention of the principle** — a reserved in-distribution slice (training-axis
  values, never trained on) and out-of-distribution dilemmas (the held-out axis
  values). Measuring both reproduces the paper's two-part claim: demonstrations
  succeed in-distribution but fail OOD, while reasoning holds up OOD.
- **Generalization distance is configurable.** `config.HELDOUT_AXIS` chooses which
  decoy axis to hold out — `domain` (near), `perspective` (far), or
  `irreversibility_type` (farthest: the principle must transfer across the *kind*
  of one-way door).
- **Judge.** A blinded Claude judge reads only (dilemma, response) — never which
  arm — and scores the recommended **action** against the answer key, after
  passing a hand-labeled gold gate.
- **Metric.** Headline is **balanced accuracy** (mean per-action recall). Labels
  are sampled 1:1 by construction, so it tracks raw accuracy closely; it guards
  the residual imbalance left after control filtering and floors any constant
  policy at 0.5 regardless. All intervals are **bootstrap over dilemma clusters**:
  the three seeds share one training set (only LoRA init and shuffle differ), so
  per-seed responses to the same dilemma are correlated replicas, not independent
  draws — resampling them individually would shrink every CI by ~√3. The
  per-`structure` breakdown is the shortcut-breaker check: a model that only
  learned "urgency → act" passes `delay_is_the_door` but fails `false_door`.

## Results

One run of `Qwen/Qwen3.5-4B` (4-bit QLoRA), 3 seeds per trained arm (`base` is a
single greedy-decode run), held-out axis = `domain` (health, relationships,
travel, legal never seen in training). Eval: 59 in-distribution and 86 OOD
dilemmas per seed, all control-clean and key-validated. The blind judge passed
the gold gate at **100% (n=45)**. Numbers are balanced accuracy (mean per-action
recall; 0.5 = any constant policy); brackets are 95% bootstrap CIs over dilemma
clusters.

**Headline — the why-arm leads on unseen domains** (`figures/headline_*`):

| arm      | in-dist | OOD  |
|----------|---------|------|
| base     | 0.68 [0.56, 0.79] | 0.81 [0.73, 0.89] |
| demos    | 0.89 [0.81, 0.95] | 0.84 [0.78, 0.89] |
| placebo  | 0.89 [0.83, 0.94] | 0.80 [0.74, 0.87] |
| **why**  | **0.94 [0.89, 0.98]** | **0.95 [0.91, 0.99]** |

Both trained arms learn the task in-distribution; on unseen domains `why` leads
and `demos`/`placebo` give most of their edge back. (Read comparisons *within* a
column: the two splits are not difficulty-matched — `base` scores higher OOD
than in-dist because the in-dist slice happens to carry twice the share of
`false_door` items, the structure the untrained model fails hardest.) Per-seed
OOD accuracy: demos 0.78/0.84/0.88, placebo 0.80/0.84/0.76, why
**0.95/0.94/0.97** — the worst `why` seed beats the best seed of either control
arm.

**It's the reasoning, not the tokens** — the OOD gap decomposed
(`figures/decomposition_*`):

| effect                      | OOD gap | 95% CI           | in-dist gap |
|-----------------------------|--------|------------------|-------------|
| length (placebo − demos)    | −0.03  | [−0.09, +0.02]   | 0.00 [−0.07, +0.07] |
| **reasoning (why − placebo)** | **+0.16** | **[+0.09, +0.23]** | +0.05 [−0.01, +0.11] |
| total (why − demos)         | +0.12  | [+0.05, +0.19]   | +0.05 [−0.02, +0.13] |

The length-matched placebo isolates the "more output" confound: it clears
nothing. The entire generalization gain is the reasoning content — and the
in-dist column doubles as a built-in control: where all arms have seen the
domains, the reasoning gap is consistent with zero. The reason only pays off
where transfer is required, which is exactly the paper's claim.

**The shortcut-breaker** — accuracy by situation structure, in-dist → OOD
(`figures/by_structure_*`):

| structure (→ correct label)          | demos | placebo | why  |
|--------------------------------------|-------|---------|------|
| delay_is_the_door (→ commit_now)     | 1.00 → 1.00 | 0.94 → 0.98 | 1.00 → 0.95 |
| false_door (→ commit_now)            | 0.94 → 1.00 | 0.96 → 1.00 | 0.89 → 1.00 |
| one_way_door_with_escape (→ reversible_first) | 0.81 → **0.67** | 0.82 → **0.62** | 0.94 → **0.94** |

This is the mechanism. `demos`/`placebo` learn the *shortcut* "when in doubt,
commit" — right on the two commit structures, and already their weak spot
in-distribution (~0.81 on the escape structure). Out of distribution the latent
weakness becomes a collapse (0.67 / 0.62) on **the one structure that needs the
principle** (recognize the reversible escape hatch and take it), while `why`
doesn't move (0.94 → 0.94; +0.32 reasoning gap on that structure OOD, CI
[0.20, 0.44]). Tables in `results/metrics/`.

**Internalized, or just thinking out loud?** With the default `reason_first`
targets the model reasons *before* acting at probe time, so the gap above could
be test-time chain-of-thought rather than an internalized principle —
Anthropic's stronger claim. The `action_first` ablation re-runs placebo/why on
the *same* dilemmas with the same content reordered so the action comes first
(content-preserving rewrite, control-validated: pass 97.6%, length ratio 0.79,
gold gate again 100%), leaving no room to reason before answering
(`figures/order_ablation_*`):

| OOD reasoning gap (why − placebo) | gap | 95% CI |
|-----------------------------------|-----|--------|
| reasoning before action           | +0.16 | [+0.09, +0.23] |
| **action before reasoning**       | **+0.06** | **[+0.02, +0.11]** |

About 40% of the generalization gain survives when the model must answer before
it can reason — smaller, but the interval still clears zero, and the per-seed
OOD scores don't overlap (why 0.89/0.88/0.89 vs placebo 0.81/0.82/0.85). On the
escape structure the action-first why arm scores 0.81 OOD vs placebo's 0.69
(reason-first: 0.94 vs 0.62). Read: part of the why-arm's edge *is* test-time
reasoning, and a real internalized remainder persists without it — both halves
of the paper's story, with the split quantified.

**Caveats specific to this run.** The `base` arm answers in a verbose instruct
style (~600-word essays; probed at a raised token budget so it is not truncated)
and is a single run — treat its row as a rough floor, not a clean principle-only
comparison. The realised `placebo/why` length ratio was 0.78 — the placebo runs
~22% shorter than `why` (length is prompted, not enforced; see
`placebo_why_length_ratio` in `control_report.json`) — which if anything makes
the length null conservative. Judge legibility is not driving the gap either:
the judge returned `unclear` (always scored wrong) for exactly 1 of 435 demos
responses and 0 of the placebo/why ones, so no arm is winning by being easier to
parse.

## Setup

```bash
uv sync                       # torch 2.6.0+cu124 (Pascal-compatible)
cp .env.example .env          # then add ANTHROPIC_API_KEY (and HF_TOKEN if gated)
```

GPU notes (GTX 1070, Pascal / sm_61): the headline model (Qwen3.5-4B) is trained
with 4-bit QLoRA in **bf16** (gradient checkpointing, batch size 1, seq-len 512,
paged 8-bit AdamW). Measured peak is ~7.1 GB of the 8 GB card, so **run the long
job headless** — the desktop (Xorg + gnome-shell) holds ~0.8–1.4 GB and can tip a
full-length sequence into OOM. Wall-clock is ~42 min/adapter, ~6 h for the full
9-adapter grid. Two Pascal-specific pins: (1) the torch index is cu124 because
later builds dropped Pascal kernels, so don't bump it; (2) training is bf16, not
fp16 — fp16 mixed precision needs a GradScaler whose unscale op has no bf16 CUDA
kernel on Pascal. The `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` allocator
is set automatically in `config`. Fall back to `--only qwen3-1.7b` if the 4-bit
wall-clock is impractical.

## Run

```bash
uv run one-way-door generate     # synthesize dilemmas + paired demos/why responses
uv run one-way-door control      # validate the paired-response control + answer key
uv run one-way-door train        # LoRA-fine-tune one adapter per (arm, seed)
uv run one-way-door probe        # recommendations on held-out domains
uv run one-way-door judge        # gold gate, then blind action-scoring
uv run one-way-door metrics      # accuracy / by-domain / by-answer-type / gap
uv run one-way-door figures      # plots
# or run the whole thing: uv run one-way-door all
```

Each stage caches to `results/` and is resumable. `generate`, `control`, and
`judge` call the Anthropic API (cost); `train` and `probe` use the GPU. `all`
runs every stage, including the API ones.

**Action-first ablation.** With the default `WHY_ORDER=reason_first`, the `why`
target reasons *before* acting, so at probe time the model can reason its way to
the answer — the gap could be test-time chain-of-thought rather than an
internalized principle. The ablation reorders the *same* dataset so the action
comes first (content-preserving rewrite; dilemmas, `demos`, and the answer key
carry over unchanged, keeping the two runs paired) and re-trains only the
order-sensitive arms into a separate results root:

```bash
export WHY_ORDER=action_first
export OWD_RESULTS_DIR=$PWD/results_action_first OWD_FIGURES_DIR=$PWD/figures_action_first
uv run one-way-door rerender     # reorder placebo/why from results/ (API)
uv run one-way-door control      # re-validate the reordered pairs (API)
uv run one-way-door train --only qwen3.5-4b --arms placebo why
uv run one-way-door probe --only qwen3.5-4b --arms placebo why
uv run one-way-door judge && uv run one-way-door metrics && uv run one-way-door figures
```

(`demos` and `base` are order-invariant; copy their probe parquets and labels
from the main run, or probe them fresh.) Then, back in the main environment,
render the side-by-side comparison figure:

```bash
uv run one-way-door figures --ablation results_action_first
```

**Cross-family judge.** The data author and the primary judge are both Claude.
`crossjudge` re-scores every cached generation with an OpenAI judge behind the
same gold gate and writes `crossjudge_report.json` plus `*_openai` metric
tables (`uv sync --extra crossjudge`, set `OPENAI_API_KEY`):

```bash
uv run one-way-door crossjudge
```

## Limitations

- **Length confound — controlled, not eliminated.** `why` is longer than `demos`,
  so the gap could be "more output" rather than "reasoning." The `placebo` arm
  (length-matched to `why`, reasoning-free) isolates this: the headline is the
  `why − placebo` reasoning gap, not the raw `why − demos` gap. This only holds if
  the placebo is genuinely length-matched — it is prompted, not enforced, and the
  realised ratio was 0.78, so the reasoning gap still carries ~22% residual
  length (in the conservative direction, given the length effect is null) — and
  if Claude never smuggles reversibility hints into the filler (the control pass
  checks `placebo_is_clean`).
- **Eval filtering trims the hardest items.** Eval dilemmas are kept only when
  the control pass re-derives the design label (`key_match`), which drops the
  most ambiguous items (~2%) and inflates *absolute* accuracy for every arm. The
  arm *gaps* — the actual claim — are unaffected, but the levels read a touch
  easier than the raw task.
- **The control pass is not fully independent.** The generator and the
  control/answer-key validator are the same model (`claude-sonnet-4-6`), so
  correlated blind spots could pass both. Only the *judge* (Haiku, a different
  model) is validated against human gold labels.
- **Decoys balanced in expectation, not exactly.** Each decoy is sampled iid
  uniform, so the within-label marginals are balanced up to sampling noise (see
  `balance.json`); they are not exactly equal. With the default counts the
  imbalance is small, but a stratified sampler would make it exact.
- **OOD is a within-task shift, not a task shift.** The paper's OOD test changed
  the task (agent-in-dilemma → advice-to-user); here train and eval are the same
  task with one decoy axis held out. `HELDOUT_AXIS=irreversibility_type` is the
  strongest version (transfer across the *kind* of one-way door), but it is still
  milder than a task-structure shift.
- **Authored demos, not on-policy.** The paper's demonstrations arm was *filtered
  from the model's own samples*; here the arms are authored matched renderings.
  That buys a cleaner control (and the placebo arm the paper lacked) but diverges
  from the self-distillation setup — the demos arm learns from text the model
  wouldn't necessarily have produced.
- **Internalization vs. chain-of-thought — measured, one caveat.** The
  `action_first` ablation (see Results) shows ~40% of the reasoning gap survives
  with no room for test-time reasoning. One asymmetry remains: the ablation's
  targets are reordered rewrites of the originals rather than independently
  authored, so `action_first` training text is slightly less natural — if
  anything a handicap for the surviving gap, but not a perfectly symmetric
  comparison.
- **One judge family.** The judge and the data author are both Claude; a
  cross-family judge would harden the headline number.
- **Small scale.** A weekend run on one 4B model; treat it as a directional echo
  of the paper's claim, not a measurement.

## Test

```bash
uv run pytest                    # pure-Python core, no GPU or API needed
uv run pytest -m gpu             # add the live training/generation checks
uv run ruff check . && uv run ruff format --check .
```

## Layout

```
src/one_way_door/
  config.py       models, LoRA spec, labels, held-out axis, sampling, env
  taxonomy.py     two-tier taxonomy + balanced scenario sampler (pure)
  data.py         dilemma / response-pair / answer-key / gold loaders
  generate.py     Claude synthesizes dilemmas + demos/placebo/why responses
                  (+ the order-preserving rewriter for the WHY_ORDER ablation)
  control.py      second-pass validator of the paired-response control
  stats.py        Wilson and bootstrap CIs, proportion tests (pure NumPy)
  metrics.py      balanced accuracy / by-structure / gap-decomposition tables
  sft.py          QLoRA training: one adapter per (arm, seed)
  model.py        base + adapter runner: blind held-out generation
  probe.py        held-out recommendations -> parquet
  judge.py        blind action-scoring judge + gold gate (+ cross-family judge)
  figures.py      plots
  pipeline.py     resumable orchestration
  smoke.py        per-model load / generate sanity check
  cli.py          `one-way-door <stage>`
data/             taxonomy.json, seed_examples.json, gold.json
tests/            pure-Python unit tests
```
