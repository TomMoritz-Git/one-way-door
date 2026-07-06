# one-way-door

A weekend reproduction, in miniature, of Anthropic's *Teaching Claude why*
finding: training a model on the **reasoning behind** a behavior generalizes
out-of-distribution better than training on the behavior alone. We teach one
small model one concrete principle — **favor reversible actions; when a choice
is a one-way door, slow down** — and ask whether it carries the principle into
domains it never trained on.

It does, and the gap survives a length-matched placebo, seed-disjoint
replication, and an action-first ablation that removes test-time reasoning.
Full write-up: [*The reason generalizes, the rule
doesn't*](https://tommoritz-git.github.io/blog/the-reason-generalizes/).

## How it works

- **Task.** A 2-class decision scored on the recommended action:
  `reversible_first` vs `commit_now`. The correct call is *not* always "wait" —
  the situation structure (one-way door with escape / delay is the door / false
  door) sets the label, so a model that always stalls fails as loudly as one
  that always acts.
- **Taxonomy.** Labels are sampled 1:1; every decoy axis (irreversibility type,
  trap, perspective, domain) is drawn independently and balanced within each
  label, so no surface cue predicts the answer (`data/taxonomy.json`, realised
  balance in `results/dataset/balance.json`).
- **Data.** Claude writes each dilemma plus one action rendered three ways in a
  single call: **demos** (bare action, ~25 words), **placebo** (action + neutral
  filler, ~91 words), **why** (action + reversibility reasoning, ~116 words).
  The placebo carries the length without the reason.
- **Control.** A second Claude pass verifies all three arms recommend the same
  action (99.8%), that demos/placebo carry no reversibility reasoning (0.99 /
  0.98 clean), and re-derives the answer key (0.98 agreement).
- **Train.** One QLoRA adapter per (arm, seed), 3 seeds, identical
  hyper-parameters; the native `<think>` scratchpad is disabled everywhere.
- **Probe.** Blind recommendations on unseen dilemmas: a reserved in-distribution
  slice and an OOD slice on four held-out domains (health, relationships,
  travel, legal). `config.HELDOUT_AXIS` picks the generalization distance.
- **Judge.** A blinded Claude judge reads only (dilemma, response) and scores
  the action, after passing a hand-labeled gold gate at **100% (n=45)**.
- **Stats.** Headline is balanced accuracy (any constant policy floors at 0.5).
  All CIs bootstrap over **dilemma clusters** — the seeds share one training
  set, so their responses are correlated replicas, not independent draws.

## Results

One run of `Qwen/Qwen3.5-4B` (4-bit QLoRA), 59 in-dist / 86 OOD dilemmas per
seed, all control-clean. Balanced accuracy:

| arm      | in-dist | OOD  |
|----------|---------|------|
| base     | 0.68    | 0.81 |
| demos    | 0.89    | 0.84 |
| placebo  | 0.89    | 0.80 |
| **why**  | **0.94**| **0.95** |

All trained arms learn the task; on unseen domains only `why` keeps its edge,
and the per-seed scores are disjoint (why 0.95/0.94/0.97 vs at best 0.88 for
either control arm). Decomposing the OOD gap: length (placebo − demos) is null
at −0.03 [−0.09, +0.02], reasoning (why − placebo) is the whole effect at
**+0.16 [+0.09, +0.23]**. In-distribution the same reasoning gap is ~0 — the
reason only pays off where transfer is required.

The mechanism is a shortcut (`figures/by_structure_*`): demos/placebo learn
"when stakes feel high, commit", which is right on two of three structures and
transfers fine. On the one structure that needs the principle they go 0.81 →
0.67 and 0.82 → 0.62 (in-dist → OOD) while `why` doesn't move (0.94 → 0.94;
reasoning gap +0.32 [0.20, 0.44] there).

Is it internalized, or test-time chain-of-thought? Retraining on the same pairs
reordered action-first (no room to reason before answering) shrinks the OOD
reasoning gap from +0.16 to **+0.06 [+0.02, +0.11]** — ~40% of the gain lives
in the weights, the rest is test-time reasoning (`figures/order_ablation_*`).

Caveats: the `base` row is a single verbose-style run (loose floor, and its
in-dist < OOD shape is split composition, not generalization); the realised
placebo/why length ratio was 0.78, which errs conservative.

## Limitations

- OOD here is a within-task domain shift, milder than the paper's task shift
  (`HELDOUT_AXIS=irreversibility_type` is the stronger version).
- Demos are authored, not filtered from the model's own samples.
- Eval keeps only key-validated items (~2% trimmed), flattering absolute levels;
  arm gaps are unaffected.
- Generator and control pass share a model family; only the judge is
  human-gold-validated. A cross-family OpenAI judge ships (`crossjudge`) but
  hasn't been run.
- The ablation's targets are reordered rewrites, not fresh authorship — if
  anything a handicap on the surviving gap.
- One 4B model, one weekend: a directional echo, not a measurement.

## Setup

```bash
uv sync                       # torch 2.6.0+cu124 (Pascal-compatible)
cp .env.example .env          # then add ANTHROPIC_API_KEY (and HF_TOKEN if gated)
```

GPU notes (GTX 1070, 8GB): 4-bit QLoRA in bf16, peak ~7.1GB — run headless (the
desktop can tip it into OOM). Two Pascal pins: keep the cu124 torch index
(later builds dropped Pascal kernels) and bf16 (fp16's GradScaler has no bf16
unscale kernel on Pascal). ~1h/adapter wall-clock.

## Run

```bash
uv run one-way-door generate     # dilemmas + demos/placebo/why (API)
uv run one-way-door control      # paired-response audit + answer key (API)
uv run one-way-door train        # one adapter per (arm, seed)
uv run one-way-door probe        # blind recommendations on eval items
uv run one-way-door judge        # gold gate, then blind action-scoring (API)
uv run one-way-door metrics      # accuracy / gap tables
uv run one-way-door figures      # plots
```

Every stage caches to `results/` and resumes. The action-first ablation reuses
the same dataset, reordered:

```bash
export WHY_ORDER=action_first OWD_RESULTS_DIR=$PWD/results_action_first
uv run one-way-door rerender && uv run one-way-door control
uv run one-way-door train --arms placebo why && uv run one-way-door probe --arms placebo why
uv run one-way-door judge && uv run one-way-door metrics
```

Then `uv run one-way-door figures --ablation results_action_first` renders the
comparison. `crossjudge` re-scores everything with an OpenAI judge
(`uv sync --extra crossjudge`, needs `OPENAI_API_KEY`).

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
  generate.py     Claude authors dilemmas + arms; rewriter for the ablation
  control.py      second-pass validator of the paired-response control
  stats.py        Wilson and bootstrap CIs (pure NumPy)
  metrics.py      balanced accuracy / by-structure / gap tables (clustered CIs)
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
