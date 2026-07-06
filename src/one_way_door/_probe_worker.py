"""Subprocess worker: probe exactly one ``(model, arm, seed)`` combo, then exit.

Run as ``python -m one_way_door._probe_worker <model_key> <arm> <seed>``.

Like training, each probe combo runs in its own process: loading a 4-bit base +
adapter ten times in a single process leaks VRAM (bitsandbytes), so the OS
reclaiming memory on exit is what keeps later combos from OOMing on the 8GB card.
The combo writes its own ``generations/<model>/<arm>-seed<seed>.parquet`` so
:func:`one_way_door.pipeline.probe_stage` stays resumable.
"""

import sys

from . import config, pipeline, probe


def main(argv: list[str] | None = None) -> int:
    """Probe one combo named by ``argv = [model_key, arm, seed]``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3:
        print("usage: _probe_worker <model_key> <arm> <seed>")
        return 2
    model_key, arm, seed = argv[0], argv[1], int(argv[2])
    spec = config.MODELS_BY_KEY[model_key]

    eval_set = pipeline.eval_dilemmas()
    model_dir = pipeline._dirs()["generations"] / spec.key
    model_dir.mkdir(parents=True, exist_ok=True)
    fpath = model_dir / f"{arm}-seed{seed}.parquet"

    df = probe.generate_for(spec, eval_set, arm, seed)
    df.to_parquet(fpath, index=False)
    trunc = float(df["truncated"].mean())
    flag = "  <-- raise PROBE_MAX_NEW_TOKENS" if trunc > 0.02 else ""
    print(
        f"[probe-worker] {model_key} {arm} seed{seed} -> {fpath} ({len(df)} rows; "
        f"{trunc:.1%} truncated){flag}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
