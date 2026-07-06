"""Command-line entry point: ``one-way-door <stage>``.

Stages mirror the pipeline and are individually resumable::

    one-way-door smoke      # verify each base model loads and generates
    one-way-door generate   # synthesize dilemmas + paired demos/why responses
    one-way-door rerender   # reorder an existing dataset to WHY_ORDER (ablation)
    one-way-door control    # independently validate the paired-response control
    one-way-door train      # LoRA-fine-tune one adapter per (arm, seed)
    one-way-door probe      # generate recommendations on held-out domains
    one-way-door judge      # gold gate + blind action-scoring vs the answer key
    one-way-door crossjudge # re-score with an OpenAI judge (cross-family check)
    one-way-door metrics    # accuracy / by-domain / by-answer-type / gap tables
    one-way-door figures    # the why-vs-demos panels
    one-way-door all        # train -> probe -> judge -> metrics -> figures
"""

import argparse
from pathlib import Path

from . import config


def _select(only: list[str] | None) -> tuple[config.ModelSpec, ...]:
    if not only:
        return config.MODELS
    chosen = []
    for key in only:
        if key not in config.MODELS_BY_KEY:
            raise SystemExit(f"unknown model key {key!r}; known: {list(config.MODELS_BY_KEY)}")
        chosen.append(config.MODELS_BY_KEY[key])
    return tuple(chosen)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested stage(s)."""
    parser = argparse.ArgumentParser(prog="one-way-door", description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    stages = (
        "smoke",
        "generate",
        "rerender",
        "control",
        "train",
        "probe",
        "judge",
        "crossjudge",
        "metrics",
        "figures",
        "all",
    )
    for name in stages:
        sp = sub.add_parser(name)
        if name in ("smoke", "train", "probe", "all"):
            sp.add_argument("--only", nargs="+", help="restrict to these model keys")
        forceable = ("generate", "rerender", "control", "train", "probe", "judge", "crossjudge")
        if name in forceable or name == "all":
            sp.add_argument("--force", action="store_true", help="recompute cached outputs")
        if name in ("train", "probe"):
            sp.add_argument(
                "--arms",
                nargs="+",
                help="restrict to these arms (probe also accepts 'base')",
            )
        if name == "rerender":
            sp.add_argument(
                "--source",
                default=str(config.ROOT / "results"),
                help="results root whose dataset to reorder (default: <root>/results)",
            )
        if name == "figures":
            sp.add_argument(
                "--ablation",
                help="results root of an action_first run; adds the order-ablation figure",
            )

    args = parser.parse_args(argv)
    config.load_env()
    force = getattr(args, "force", False)

    if args.stage == "smoke":
        from . import smoke

        results = smoke.run(_select(args.only))
        return 0 if all(r.ok for r in results) else 1

    from . import pipeline

    arms = tuple(a) if (a := getattr(args, "arms", None)) else None
    if args.stage in ("generate", "all"):
        pipeline.generate_stage(force=force)
    if args.stage == "rerender":
        pipeline.rerender_stage(Path(args.source), force=force)
    if args.stage in ("control", "all"):
        pipeline.control_stage(force=force)
    if args.stage in ("train", "all"):
        pipeline.train_stage(_select(getattr(args, "only", None)), force=force, arms=arms)
    if args.stage in ("probe", "all"):
        pipeline.probe_stage(_select(getattr(args, "only", None)), force=force, arms=arms)
    if args.stage in ("judge", "all"):
        pipeline.judge_stage(force=force)
    if args.stage == "crossjudge":
        pipeline.crossjudge_stage(force=force)
    if args.stage in ("metrics", "all"):
        pipeline.metrics_stage()
    if args.stage in ("figures", "all"):
        from . import figures

        ablation = getattr(args, "ablation", None)
        for path in figures.make_all(Path(ablation) if ablation else None):
            print(f"[figures] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
