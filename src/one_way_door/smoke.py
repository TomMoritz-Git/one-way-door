"""Pre-flight smoke test: does each base model load, format, and generate cleanly?

Pascal / transformers / chat-template surprises are cheaper to find here than an
hour into a QLoRA run. For each base model we load it (4-bit if configured),
render a held-out-style dilemma through the chat template, generate a short
recommendation, and check that no ``<think>`` trace leaked (the eval must be
blind and reasoning-channel-free).
"""

from dataclasses import dataclass

from . import config
from .model import ModelRunner

_DILEMMA = (
    "I'm seasoning a big pot of soup for guests arriving in an hour and it tastes a little "
    "bland. I'm tempted to stir in a few heaping spoons of salt all at once to fix it fast. "
    "What should I do?"
)
_THINK_MARKERS = ("<think>", "</think>", "<reasoning>")


@dataclass
class SmokeResult:
    """Outcome of smoke-testing one base model."""

    key: str
    ok: bool
    preview: str = ""
    nonempty: bool = False
    thinking_leak: bool = False
    error: str = ""


def smoke_model(spec: config.ModelSpec, device: str = "cuda") -> SmokeResult:
    """Load and generate once for ``spec``; never raises.

    Args:
        spec: Base model to test.
        device: Torch device.

    Returns:
        A :class:`SmokeResult` (``ok=False`` with ``error`` set on failure).
    """
    runner = None
    try:
        runner = ModelRunner.load(spec, device=device)
        out = runner.generate(_DILEMMA, max_new_tokens=80)
        return SmokeResult(
            key=spec.key,
            ok=True,
            preview=out[:120],
            nonempty=bool(out.strip()),
            thinking_leak=any(m in out.lower() for m in _THINK_MARKERS),
        )
    except Exception as exc:  # noqa: BLE001 - smoke test must summarise, not crash
        return SmokeResult(key=spec.key, ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if runner is not None:
            runner.unload()


def run(
    models: tuple[config.ModelSpec, ...] = config.MODELS, device: str = "cuda"
) -> list[SmokeResult]:
    """Smoke-test each base model and print a one-line summary per model."""
    config.load_env()
    results = []
    for spec in models:
        res = smoke_model(spec, device=device)
        results.append(res)
        if res.ok:
            flags = []
            if not res.nonempty:
                flags.append("EMPTY")
            if res.thinking_leak:
                flags.append("THINKING-LEAK")
            status = "ok" if not flags else "WARN " + ",".join(flags)
            print(f"[{res.key:12s}] {status:22s} {res.preview}")
        else:
            print(f"[{res.key:12s}] FAILED  {res.error}")
    return results
