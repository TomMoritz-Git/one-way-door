"""Static configuration: models, LoRA spec, paths, grids, and environment loading.

Everything a human might want to tweak between runs lives here as plain data. No
module-level side effects beyond defining constants (``load_env`` is explicit).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Use the expandable-segments CUDA allocator so the large fp32 LM-head logits
# (vocab-sized) don't fail on contiguous-block fragmentation on the 8GB 1070.
# Set at import, before any torch CUDA init in this process. setdefault so an
# explicit user override still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- Paths -----------------------------------------------------------------
# Repo root is two parents up from this file: <root>/src/one_way_door/config.py
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
# OWD_RESULTS_DIR / OWD_FIGURES_DIR redirect a whole run into a separate root --
# used for ablations (e.g. the action-first rerun) so they never clobber the
# main results. Read at import: set them in the shell, not .env.
RESULTS_DIR = Path(os.environ.get("OWD_RESULTS_DIR", str(ROOT / "results")))
FIGURES_DIR = Path(os.environ.get("OWD_FIGURES_DIR", str(ROOT / "figures")))

TAXONOMY_PATH = DATA_DIR / "taxonomy.json"
SEED_EXAMPLES_PATH = DATA_DIR / "seed_examples.json"
GOLD_PATH = DATA_DIR / "gold.json"


# --- Models ----------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """A base model to fine-tune (one LoRA adapter per training arm).

    Attributes:
        key: Short slug used in filenames and figures.
        hf_id: Hugging Face repository id.
        provider: Organisation that trained the model.
        params_b: Parameter count in billions (approximate).
        gated: Whether the HF repo requires accepting a license / token.
        chat_template_kwargs: Extra kwargs passed to ``apply_chat_template``
            (e.g. ``{"enable_thinking": False}`` to disable Qwen's <think> mode).
        system: Minimal system message, used *only* to disable thinking traces;
            never carries the reversibility principle (the eval must be blind).
        load_in_4bit: Load with bitsandbytes NF4 (QLoRA) -- required for the 4B
            headline model on an 8GB card.
        attn_implementation: ``"sdpa"`` (math backend, stable in fp16 on Pascal);
            ``"eager"`` only where a model needs it for correctness.
        notes: Free-form notes.
    """

    key: str
    hf_id: str
    provider: str
    params_b: float
    gated: bool = False
    chat_template_kwargs: dict[str, object] = field(default_factory=dict)
    system: str | None = None
    load_in_4bit: bool = False
    attn_implementation: str = "sdpa"
    notes: str = ""


# The native <think> scratchpad is disabled for every model so that the only
# reasoning in play is what we train into the why-arm's visible answer -- any
# held-out gap is then attributable to the training data, not inherited CoT.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="qwen3.5-4b",
        hf_id="Qwen/Qwen3.5-4B",
        provider="Alibaba",
        params_b=4.0,
        chat_template_kwargs={"enable_thinking": False},
        load_in_4bit=True,
        notes="Headline. QLoRA (4-bit NF4) to fit 8GB; <think> off both arms.",
    ),
    ModelSpec(
        key="qwen3-1.7b",
        hf_id="Qwen/Qwen3-1.7B",
        provider="Alibaba",
        params_b=1.7,
        chat_template_kwargs={"enable_thinking": False},
        notes="Robustness rerun / fp16 fallback if 4B wall-clock is impractical.",
    ),
)

MODELS_BY_KEY: dict[str, ModelSpec] = {m.key: m for m in MODELS}


# --- Training arms ---------------------------------------------------------
# The arms are identical but for the target completion text, all recommending the
# same action:
#   demos   -- the bare action (short, no reasoning)
#   placebo -- the action padded to ~the length of `why` with neutral on-topic
#              filler, but NO reversibility reasoning (long, no reasoning)
#   why     -- the action plus the reversibility reasoning (long, +reasoning)
# This 2x2-minus-one lets us decompose the demos->why gap: demos->placebo isolates
# the effect of longer output, placebo->why isolates the effect of the reasoning
# content (the variable we actually care about).
ARMS: tuple[str, ...] = ("demos", "placebo", "why")

# Where the reasoning sits in the `why` target (and, in parallel, where the filler
# sits in `placebo`, so the action stays in the same position across the two long
# arms). "reason_first" puts the reversibility reasoning before the action, so it
# is causally upstream of the action -- the natural, stronger structure (the model
# can re-derive the answer, and training conditions the action on the reasoning).
# "action_first" emits the action before any reasoning, a stricter test of pure
# internalization (a gap then cannot come from test-time chain-of-thought) but a
# weaker training signal. It is an ablation knob (env-overridable; read at
# import, so set it in the shell, not .env).
WHY_ORDER = os.environ.get("WHY_ORDER", "reason_first")  # "reason_first" | "action_first"


@dataclass(frozen=True)
class LoRASpec:
    """LoRA / QLoRA + SFT hyper-parameters (shared identically across arms)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 16
    # 512 covers the longest prompt+completion (~360 tok incl. a long `why`) with
    # margin -- so the reason-first action at the end is never truncated -- while
    # halving the vocab-sized fp32 logit memory vs 1024 (the 8GB OOM lever).
    max_seq_len: int = 512
    warmup_ratio: float = 0.03


LORA = LoRASpec()

# Random seeds: each (arm, seed) is one adapter; the gap is averaged over seeds.
SEEDS: tuple[int, ...] = (0, 1, 2)

# Held-out probe: greedy decode for determinism. Trained arms emit a terse answer
# and hit EOS well under ~150 tokens (so they are invariant to this cap), but the
# UNTRAINED base answers in a verbose instruct style (~300+ words) -- it needs a
# generous budget to finish, otherwise its recommendation is truncated and the
# judge can't score it fairly. Sized for the base; harmless for the trained arms.
PROBE_MAX_NEW_TOKENS = 2048


# --- Labels and the answer key ---------------------------------------------
# The task is a 2-class decision, scored on the action the response recommends.
# The blind judge emits one of JUDGE_ACTIONS; the two real classes are LABELS,
# and ``unclear`` (no actionable recommendation) is always scored wrong.
LABELS: tuple[str, ...] = ("reversible_first", "commit_now")
JUDGE_ACTIONS: tuple[str, ...] = (*LABELS, "unclear")

# The correct label is set by Tier 1 of the taxonomy (the situation structure);
# the mapping lives in taxonomy.json and is exposed via
# ``taxonomy.Taxonomy.action_for``. Everything else in the taxonomy is a balanced
# decoy, so no surface feature predicts the label.

# --- Generalization split --------------------------------------------------
# Which decoy axis to hold out for the out-of-distribution eval, and which of its
# values to reserve. Pick the generalization "distance": "domain" (near),
# "perspective" (far), or "irreversibility_type" (farthest -- the principle must
# transfer across the *kind* of one-way door). HELDOUT_VALUES must be values of
# the chosen axis; the rest are used for training + the in-distribution eval.
HELDOUT_AXIS = "domain"
HELDOUT_VALUES: tuple[str, ...] = ("health", "relationships", "travel", "legal")

# --- Dataset sampling ------------------------------------------------------
# Scenarios are sampled (not enumerated): 50/50 across the two labels, with every
# decoy axis drawn independently and uniformly so it is balanced within each
# label. Counts are PER LABEL. role="train" feeds SFT; role="eval" is probed and
# never trained -- "in_dist" reuses training-axis values (unseen items), "ood"
# uses the held-out values.
N_TRAIN_PER_LABEL = 150
N_INDIST_EVAL_PER_LABEL = 30
N_OOD_EVAL_PER_LABEL = 45
SAMPLER_SEED = 0


# --- Anthropic models ------------------------------------------------------
# A capable model authors the dataset; a cheap one judges at volume.
GEN_MODEL = "claude-sonnet-4-6"
CONTROL_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-haiku-4-5"
# Cross-family judge (OpenAI): same prompt/tool/gold gate, different model family;
# needs OPENAI_API_KEY and the `crossjudge` extra.
CROSS_JUDGE_MODEL = os.environ.get("CROSS_JUDGE_MODEL", "gpt-5-mini")
ANTHROPIC_TEMPERATURE = 0.0
GEN_TEMPERATURE = 0.8  # some diversity when authoring dilemmas
JUDGE_AGREEMENT_THRESHOLD = 0.95


def load_env(dotenv_path: Path | None = None) -> None:
    """Load ``.env`` from the repo root into ``os.environ`` (idempotent).

    Existing environment variables win over the file, matching standard
    ``python-dotenv`` behaviour.

    Args:
        dotenv_path: Optional explicit path; defaults to ``<root>/.env``.
    """
    load_dotenv(dotenv_path or (ROOT / ".env"), override=False)


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear error.

    Args:
        name: Variable name.

    Returns:
        The variable's value.

    Raises:
        RuntimeError: If the variable is unset or empty.
    """
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to {ROOT / '.env'} and call config.load_env()."
        )
    return value
