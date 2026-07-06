"""Resumable orchestration: generate -> control -> train -> probe -> judge -> metrics.

Artifacts under ``results/``:

* ``dataset/dilemmas.parquet`` -- every synthesized dilemma (all roles/splits).
* ``dataset/pairs.parquet`` -- the demos/placebo/why responses for each dilemma.
* ``dataset/answer_key.json`` -- eval dilemma id -> distribution / structure / action.
* ``dataset/balance.json`` -- realised decoy balance within each label.
* ``dataset/control.parquet`` + ``control_report.json`` -- the paired-response audit.
* ``adapters/<key>/<arm>-seed<k>/`` -- one trained LoRA adapter each.
* ``generations/<key>/<arm>-seed<k>.parquet`` and joined ``generations.parquet``.
* ``labels.parquet`` -- judged action per generation (incremental).
* ``gold_report.json`` -- judge validation against the gold set.
* ``metrics/*.parquet`` -- accuracy and gap tables.

Each stage skips work whose output already exists (``force=True`` to redo), so a
run interrupted by an OOM, a flaky API call, or a gated model resumes cleanly.
"""

import json
from pathlib import Path

import pandas as pd

from . import config, data, metrics, taxonomy
from .data import Dilemma, ResponsePair


def _dirs() -> dict[str, Path]:
    base = config.RESULTS_DIR
    paths = {
        "base": base,
        "dataset": base / "dataset",
        "adapters": base / "adapters",
        "generations": base / "generations",
        "metrics": base / "metrics",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# --- dataset I/O -----------------------------------------------------------
def load_dataset() -> tuple[list[Dilemma], dict[str, ResponsePair]]:
    """Load the synthesized dataset from ``results/dataset``."""
    paths = _dirs()
    dpath = paths["dataset"] / "dilemmas.parquet"
    ppath = paths["dataset"] / "pairs.parquet"
    if not dpath.exists() or not ppath.exists():
        raise FileNotFoundError("no dataset; run the generate stage first")
    dilemmas = [data.dilemma_from_record(r) for r in pd.read_parquet(dpath).to_dict("records")]
    pairs = {
        r["dilemma_id"]: data.pair_from_record(r) for r in pd.read_parquet(ppath).to_dict("records")
    }
    return dilemmas, pairs


# --- generate --------------------------------------------------------------
def generate_stage(force: bool = False) -> Path:
    """Synthesize every dilemma + paired responses and cache the dataset."""
    from .generate import Generator

    paths = _dirs()
    dpath = paths["dataset"] / "dilemmas.parquet"
    if dpath.exists() and not force:
        print(f"[generate] dataset exists at {dpath}; use --force to regenerate")
        return dpath

    tax = taxonomy.load_taxonomy()
    seed_examples = data.load_seed_examples()
    scenarios = taxonomy.sample_scenarios(tax)
    print(
        f"[generate] held-out {config.HELDOUT_AXIS}={list(config.HELDOUT_VALUES)}; "
        f"authoring {len(scenarios)} dilemmas"
    )

    gen = Generator(tax, seed_examples)
    raw = gen.generate_many(scenarios)

    dilemma_records, pair_records, answer_key = [], [], {}
    dropped = 0
    for sc, g in zip(scenarios, raw, strict=True):
        if g is None:
            dropped += 1
            continue
        item = Dilemma(
            id=sc.uid,
            domain=sc.domain,
            split=sc.split,
            role=sc.role,
            structure=sc.structure,
            label=sc.action,
            perspective=sc.perspective,
            irreversibility_type=sc.irreversibility,
            trap=sc.trap,
            reversible_move=g.reversible_move,
            text=g.text,
            action_text=g.action_text,
        )
        dilemma_records.append(data.dilemma_to_record(item))
        pair_records.append(
            data.pair_to_record(
                ResponsePair(dilemma_id=sc.uid, demos=g.demos, placebo=g.placebo, why=g.why)
            )
        )
        if sc.role == "eval":
            answer_key[item.id] = {
                "distribution": item.distribution,
                "structure": item.structure,
                "correct_action": item.correct_action,
            }
    if dropped:
        print(f"[generate] WARNING: {dropped}/{len(scenarios)} dilemmas failed; continuing")

    pd.DataFrame(dilemma_records).to_parquet(dpath, index=False)
    pd.DataFrame(pair_records).to_parquet(paths["dataset"] / "pairs.parquet", index=False)
    (paths["dataset"] / "answer_key.json").write_text(json.dumps(answer_key, indent=2))
    # Document the realised decoy balance (no surface axis should predict the label).
    (paths["dataset"] / "balance.json").write_text(
        json.dumps(taxonomy.balance_report(scenarios), indent=2)
    )
    n_train = sum(r["role"] == "train" for r in dilemma_records)
    n_eval = len(dilemma_records) - n_train
    print(
        f"[generate] wrote {len(dilemma_records)} dilemmas "
        f"({n_train} train / {n_eval} eval) -> {dpath}"
    )
    return dpath


# --- rerender (WHY_ORDER ablation) ------------------------------------------
def rerender_stage(source: Path, force: bool = False) -> Path:
    """Clone a dataset with placebo/why reordered to the configured ``WHY_ORDER``.

    Reads the authored dataset under ``source`` (a results root from a previous
    generate run) and writes into the *current* ``config.RESULTS_DIR`` a copy in
    which each pair's ``placebo``/``why`` are reordered by :class:`generate.Rewriter`
    while dilemmas, ``demos``, and the answer key are carried over unchanged. Run
    with ``OWD_RESULTS_DIR`` pointing at a fresh root and ``WHY_ORDER`` set to the
    ablation order; the downstream stages (control -> train -> probe -> judge ->
    metrics) then run unmodified in that root.
    """
    from .generate import Rewriter

    paths = _dirs()
    if paths["dataset"].resolve() == (source / "dataset").resolve():
        raise SystemExit(
            "rerender would overwrite its own source; set OWD_RESULTS_DIR to a fresh root"
        )
    ppath = paths["dataset"] / "pairs.parquet"
    if ppath.exists() and not force:
        print(f"[rerender] dataset exists at {ppath}; use --force to redo")
        return ppath

    src_dataset = source / "dataset"
    dilemmas = pd.read_parquet(src_dataset / "dilemmas.parquet")
    pairs = pd.read_parquet(src_dataset / "pairs.parquet").set_index("dilemma_id")
    print(f"[rerender] reordering {len(pairs)} pairs to WHY_ORDER={config.WHY_ORDER!r}")

    items, ids = [], []
    for d in dilemmas.itertuples():
        p = pairs.loc[d.id]
        items.append((d.text, d.action_text, p["placebo"], p["why"]))
        ids.append(d.id)
    rewritten = Rewriter().rewrite_many(items)

    records, dropped = [], 0
    for did, rw in zip(ids, rewritten, strict=True):
        if rw is None:
            dropped += 1
            continue
        records.append(
            {
                "dilemma_id": did,
                "demos": pairs.loc[did, "demos"],
                "placebo": rw["placebo"],
                "why": rw["why"],
            }
        )
    if dropped:
        print(f"[rerender] WARNING: {dropped}/{len(ids)} rewrites failed; those pairs dropped")

    kept = {r["dilemma_id"] for r in records}
    dilemmas[dilemmas["id"].isin(kept)].to_parquet(
        paths["dataset"] / "dilemmas.parquet", index=False
    )
    pd.DataFrame(records).to_parquet(ppath, index=False)
    for name in ("answer_key.json", "balance.json"):
        (paths["dataset"] / name).write_text((src_dataset / name).read_text())
    print(f"[rerender] wrote {len(records)} pairs -> {ppath}")
    return ppath


# --- control ---------------------------------------------------------------
def control_stage(force: bool = False) -> Path:
    """Independently validate the paired-response control and the answer key."""
    from .control import Controller

    paths = _dirs()
    cpath = paths["dataset"] / "control.parquet"
    if cpath.exists() and not force:
        print("[control] control.parquet exists; use --force to redo")
        return cpath

    dilemmas, pairs = load_dataset()
    controller = Controller()
    items = [(d.text, pairs[d.id].demos, pairs[d.id].placebo, pairs[d.id].why) for d in dilemmas]
    print(f"[control] validating {len(items)} dilemmas")
    validations = controller.validate_many(items)

    rows = []
    dropped = 0
    for d, v in zip(dilemmas, validations, strict=True):
        if v is None:
            dropped += 1
            continue
        rows.append(
            {
                "dilemma_id": d.id,
                "split": d.split,
                "role": d.role,
                "distribution": d.distribution,
                "structure": d.structure,
                "same_action": v.same_action,
                "demos_is_clean": v.demos_is_clean,
                "placebo_is_clean": v.placebo_is_clean,
                "why_explains": v.why_explains,
                "control_ok": v.control_ok,
                "derived_action": v.correct_action,
                "key_match": v.correct_action == d.correct_action,
            }
        )
    if dropped:
        print(f"[control] WARNING: {dropped} dilemmas failed validation; treated as unusable")
    df = pd.DataFrame(rows)
    df.to_parquet(cpath, index=False)

    # Length-match diagnostic: the placebo only deconfounds length if it is in fact
    # about as long as `why`. Report mean word counts per arm and the ratio.
    def _mean_words(arm: str) -> float:
        counts = [len(pairs[d.id].target(arm).split()) for d in dilemmas]
        return sum(counts) / len(counts) if counts else 0.0

    lens = {arm: _mean_words(arm) for arm in config.ARMS}

    report = {
        "n": int(len(df)),
        "same_action_rate": float(df["same_action"].mean()),
        "demos_clean_rate": float(df["demos_is_clean"].mean()),
        "placebo_clean_rate": float(df["placebo_is_clean"].mean()),
        "why_explains_rate": float(df["why_explains"].mean()),
        "control_pass_rate": float(df["control_ok"].mean()),
        "mean_words": lens,
        "placebo_why_length_ratio": (lens["placebo"] / lens["why"]) if lens["why"] else None,
        "answer_key_agreement": float(df["key_match"].mean()),
        "train_usable": int(((df["role"] == "train") & df["control_ok"]).sum()),
        "eval_indist_usable": int(
            (
                (df["role"] == "eval")
                & (df["distribution"] == "in_dist")
                & df["control_ok"]
                & df["key_match"]
            ).sum()
        ),
        "eval_ood_usable": int(
            (
                (df["role"] == "eval")
                & (df["distribution"] == "ood")
                & df["control_ok"]
                & df["key_match"]
            ).sum()
        ),
    }
    (paths["base"] / "control_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"[control] control pass {report['control_pass_rate']:.1%}, "
        f"answer-key agreement {report['answer_key_agreement']:.1%} -> control_report.json"
    )
    return cpath


def _control() -> pd.DataFrame:
    cpath = _dirs()["dataset"] / "control.parquet"
    if not cpath.exists():
        raise FileNotFoundError("no control.parquet; run the control stage first")
    return pd.read_parquet(cpath)


def training_dilemmas() -> list[Dilemma]:
    """Train-role dilemmas (SFT set) that passed the paired-response control."""
    dilemmas, _ = load_dataset()
    ctrl = _control().set_index("dilemma_id")
    out = []
    for d in dilemmas:
        if d.role != "train" or d.id not in ctrl.index:
            continue
        if bool(ctrl.loc[d.id, "control_ok"]):
            out.append(d)
    return out


def eval_dilemmas() -> list[Dilemma]:
    """Eval-role dilemmas (in-dist + OOD) with a control-clean, key-validated answer.

    These are never trained on. Each carries ``distribution`` (``in_dist`` for
    train-domain items, ``ood`` for held-out ones) so the probe can show the
    in-distribution-to-OOD drop per arm.
    """
    dilemmas, _ = load_dataset()
    ctrl = _control().set_index("dilemma_id")
    out = []
    for d in dilemmas:
        if d.role != "eval" or d.id not in ctrl.index:
            continue
        row = ctrl.loc[d.id]
        if bool(row["control_ok"]) and bool(row["key_match"]):
            out.append(d)
    return out


# --- train -----------------------------------------------------------------
def train_stage(
    specs: tuple[config.ModelSpec, ...],
    force: bool = False,
    arms: tuple[str, ...] | None = None,
) -> list[Path]:
    """Train one LoRA adapter per (spec, arm, seed) on control-passing train data.

    Each adapter trains in a fresh subprocess (``one_way_door._train_worker``) so
    the OS reclaims all VRAM on exit -- bitsandbytes 4-bit + TRL leak GPU memory
    across successive in-process trainings, which OOMs every adapter after the
    first on an 8GB card. Resumable: an adapter whose directory already exists is
    skipped unless ``force``. ``arms`` restricts training to a subset (used by
    the WHY_ORDER ablation, where the demos arm is order-invariant).
    """
    import subprocess
    import sys

    from . import sft

    train = training_dilemmas()
    print(f"[train] {len(train)} usable train dilemmas")
    out = []
    for spec in specs:
        for arm in arms or config.ARMS:
            for seed in config.SEEDS:
                adir = sft.adapter_dir(spec.key, arm, seed)
                if adir.exists() and not force:
                    print(f"[train] {spec.key} {arm} seed{seed} -> cached, skipping")
                    out.append(adir)
                    continue
                cmd = [sys.executable, "-m", "one_way_door._train_worker", spec.key, arm, str(seed)]
                if force:
                    cmd.append("--force")
                print(f"[train] {spec.key} {arm} seed{seed}: launching subprocess")
                rc = subprocess.run(cmd).returncode
                if rc == 0 and adir.exists():
                    out.append(adir)
                    print(f"[train] {spec.key} {arm} seed{seed} -> {adir}")
                else:
                    print(f"[train] {spec.key} {arm} seed{seed} FAILED (rc={rc}); continuing")
    return out


# --- probe -----------------------------------------------------------------
def _probe_arms(arms: tuple[str, ...] | None = None) -> list[tuple[str, int]]:
    """(arm, seed) combinations to probe: base once, trained arms per seed."""
    wanted = arms or ("base", *config.ARMS)
    combos: list[tuple[str, int]] = []
    if "base" in wanted:
        combos.append(("base", config.SEEDS[0]))
    for arm in config.ARMS:
        if arm in wanted:
            combos += [(arm, seed) for seed in config.SEEDS]
    return combos


def probe_stage(
    specs: tuple[config.ModelSpec, ...],
    force: bool = False,
    arms: tuple[str, ...] | None = None,
) -> Path:
    """Generate recommendations on in-distribution + OOD eval items, per (spec, arm, seed).

    Each combo runs in a fresh subprocess (``one_way_door._probe_worker``) for the
    same reason as training: loading a 4-bit base + adapter many times in one
    process leaks VRAM and OOMs later combos on an 8GB card. Resumable: a combo
    whose parquet already exists is skipped unless ``force``. ``arms`` restricts
    the probe (may include ``"base"``).
    """
    import subprocess
    import sys

    paths = _dirs()
    eval_set = eval_dilemmas()
    n_ood = sum(d.distribution == "ood" for d in eval_set)
    print(
        f"[probe] {len(eval_set)} usable eval dilemmas ({len(eval_set) - n_ood} in-dist / "
        f"{n_ood} OOD)"
    )
    for spec in specs:
        model_dir = paths["generations"] / spec.key
        model_dir.mkdir(parents=True, exist_ok=True)
        for arm, seed in _probe_arms(arms):
            fpath = model_dir / f"{arm}-seed{seed}.parquet"
            if fpath.exists() and not force:
                print(f"[probe] {spec.key} {arm} seed{seed} -> cached, skipping")
                continue
            cmd = [sys.executable, "-m", "one_way_door._probe_worker", spec.key, arm, str(seed)]
            print(f"[probe] {spec.key} {arm} seed{seed}: launching subprocess")
            rc = subprocess.run(cmd).returncode
            if rc != 0 or not fpath.exists():
                print(f"[probe] {spec.key} {arm} seed{seed} FAILED (rc={rc}); continuing")
    return _join_generations()


def _join_generations() -> Path:
    paths = _dirs()
    parts = sorted(paths["generations"].glob("*/*.parquet"))
    if not parts:
        raise FileNotFoundError("no generations; run the probe stage first")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out = paths["base"] / "generations.parquet"
    df.to_parquet(out, index=False)
    _report_truncation(df, paths["base"] / "probe_truncation.json")
    return out


def _report_truncation(df: pd.DataFrame, path: Path) -> None:
    """Log + persist the budget-hit rate by arm x distribution.

    Under ``WHY_ORDER="reason_first"`` the scored action is last, so a truncated
    response can drop it: a non-trivial OOD ``why`` rate means ``PROBE_MAX_NEW_TOKENS``
    is starving exactly the cases the experiment measures.
    """
    if "truncated" not in df.columns:
        return
    rate = (
        df.groupby(["arm", "distribution"])["truncated"].mean().reset_index(name="truncated_rate")
    )
    path.write_text(json.dumps(rate.to_dict("records"), indent=2))
    overall = float(df["truncated"].mean())
    worst = rate.sort_values("truncated_rate", ascending=False).iloc[0]
    print(
        f"[probe] truncation: {overall:.1%} overall; worst "
        f"{worst['arm']}/{worst['distribution']} = {worst['truncated_rate']:.1%}"
        + ("  <-- raise PROBE_MAX_NEW_TOKENS" if worst["truncated_rate"] > 0.02 else "")
    )


# --- judge -----------------------------------------------------------------
def judge_stage(force: bool = False, gold_threshold: float | None = None) -> pd.DataFrame:
    """Validate the judge on gold, then score every not-yet-judged generation."""
    from .judge import Judge

    paths = _dirs()
    labels_path = paths["base"] / "labels.parquet"
    generations = pd.read_parquet(_join_generations())

    existing = (
        pd.read_parquet(labels_path)
        if labels_path.exists() and not force
        else pd.DataFrame(columns=["row_id", "judge_action", "rationale"])
    )
    todo = generations[~generations["row_id"].isin(set(existing["row_id"]))]
    if todo.empty:
        print(f"[judge] all {len(generations)} generations already judged; nothing to do")
        return existing

    judge = Judge()
    gold = data.load_gold()
    report = judge.validate_on_gold(
        gold, threshold=gold_threshold or config.JUDGE_AGREEMENT_THRESHOLD
    )
    (paths["base"] / "gold_report.json").write_text(
        json.dumps(
            {
                "n": report.n,
                "agreement": report.agreement,
                "passed": report.passed,
                "recall": {a: report.recall(a) for a in config.JUDGE_ACTIONS},
                "confusion": [
                    {"gold": g, "judge": j, "count": c}
                    for (g, j), c in sorted(report.confusion.items())
                ],
                "disagreements": [
                    {"id": i, "gold": g, "judge": j} for i, g, j in report.disagreements
                ],
            },
            indent=2,
        )
    )
    print(
        f"[judge] gold agreement {report.agreement:.1%} (n={report.n}) "
        f"-> {'PASS' if report.passed else 'FAIL'}"
    )
    if not report.passed:
        raise RuntimeError(
            f"judge failed gold gate: {report.agreement:.1%} < "
            f"{gold_threshold or config.JUDGE_AGREEMENT_THRESHOLD:.0%}. See gold_report.json."
        )

    print(f"[judge] scoring {len(todo)} generations ({len(existing)} cached)")
    judgments = judge.judge_many(list(zip(todo["dilemma"], todo["response"], strict=True)))
    new_labels = pd.DataFrame(
        {
            "row_id": todo["row_id"].to_numpy(),
            "judge_action": [j.action for j in judgments],
            "rationale": [j.rationale for j in judgments],
        }
    )
    labels = pd.concat([existing, new_labels], ignore_index=True)
    labels.to_parquet(labels_path, index=False)
    return labels


# --- cross-family judge -----------------------------------------------------
def crossjudge_stage(force: bool = False) -> pd.DataFrame:
    """Re-score every generation with the OpenAI judge and report agreement.

    Same gold gate, same blind (dilemma, response) input as :func:`judge_stage`;
    only the judge's model family differs. Writes ``labels_openai.parquet``,
    ``gold_report_openai.json``, ``crossjudge_report.json``, and a
    ``balanced_by_distribution_openai`` metrics table so the headline can be
    read under a judge the data author is unrelated to.
    """
    from .judge import OpenAIJudge

    paths = _dirs()
    labels_path = paths["base"] / "labels_openai.parquet"
    generations = pd.read_parquet(_join_generations())

    existing = (
        pd.read_parquet(labels_path)
        if labels_path.exists() and not force
        else pd.DataFrame(columns=["row_id", "judge_action", "rationale"])
    )
    todo = generations[~generations["row_id"].isin(set(existing["row_id"]))]

    judge = OpenAIJudge()
    if not todo.empty:
        gold = data.load_gold()
        report = judge.validate_on_gold(gold)
        (paths["base"] / "gold_report_openai.json").write_text(
            json.dumps(
                {
                    "model": judge.model,
                    "n": report.n,
                    "agreement": report.agreement,
                    "passed": report.passed,
                    "recall": {a: report.recall(a) for a in config.JUDGE_ACTIONS},
                    "disagreements": [
                        {"id": i, "gold": g, "judge": j} for i, g, j in report.disagreements
                    ],
                },
                indent=2,
            )
        )
        print(
            f"[crossjudge] {judge.model} gold agreement {report.agreement:.1%} "
            f"(n={report.n}) -> {'PASS' if report.passed else 'FAIL'}"
        )
        if not report.passed:
            raise RuntimeError(
                f"cross judge failed gold gate: {report.agreement:.1%}. "
                "See gold_report_openai.json."
            )
        print(f"[crossjudge] scoring {len(todo)} generations ({len(existing)} cached)")
        judgments = judge.judge_many(list(zip(todo["dilemma"], todo["response"], strict=True)))
        new_labels = pd.DataFrame(
            {
                "row_id": todo["row_id"].to_numpy(),
                "judge_action": [j.action for j in judgments],
                "rationale": [j.rationale for j in judgments],
            }
        )
        existing = pd.concat([existing, new_labels], ignore_index=True)
        existing.to_parquet(labels_path, index=False)
    else:
        print(f"[crossjudge] all {len(generations)} generations already judged")
    labels = existing

    # Judge-to-judge agreement (per arm) + the headline under the cross judge.
    claude = pd.read_parquet(paths["base"] / "labels.parquet")[["row_id", "judge_action"]]
    both = (
        generations[["row_id", "arm", "distribution"]]
        .merge(claude.rename(columns={"judge_action": "claude_action"}), on="row_id")
        .merge(labels[["row_id", "judge_action"]], on="row_id")
    )
    both["agree"] = both["claude_action"] == both["judge_action"]
    merged = generations.merge(labels[["row_id", "judge_action"]], on="row_id")
    merged["correct"] = merged["judge_action"] == merged["correct_action"]
    bal = metrics.balanced_accuracy(merged, ["model", "arm", "distribution"])
    bal.to_parquet(paths["metrics"] / "balanced_by_distribution_openai.parquet", index=False)
    gap = metrics.gap(merged, "why", "placebo", breakdown="distribution")
    gap.to_parquet(paths["metrics"] / "gap_reasoning_by_distribution_openai.parquet", index=False)

    report = {
        "model": judge.model,
        "n": int(len(both)),
        "judge_agreement": float(both["agree"].mean()),
        "judge_agreement_by_arm": {
            arm: float(sub["agree"].mean()) for arm, sub in both.groupby("arm")
        },
    }
    (paths["base"] / "crossjudge_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"[crossjudge] judge-to-judge agreement {report['judge_agreement']:.1%} "
        f"(n={report['n']}) -> crossjudge_report.json"
    )
    return labels


# --- metrics ---------------------------------------------------------------
def metrics_stage() -> dict[str, pd.DataFrame]:
    """Join generations with judged actions and write the accuracy/gap tables."""
    paths = _dirs()
    generations = pd.read_parquet(paths["base"] / "generations.parquet")
    labels = pd.read_parquet(paths["base"] / "labels.parquet")
    merged = generations.merge(labels[["row_id", "judge_action"]], on="row_id", how="left")

    missing = int(merged["judge_action"].isna().sum())
    if missing:
        print(f"[metrics] WARNING: {missing} generations have no judge label; dropping them")
        merged = merged[merged["judge_action"].notna()].copy()
    merged["correct"] = merged["judge_action"] == merged["correct_action"]
    ood = merged[merged["distribution"] == "ood"]

    # The headline lives in by_distribution / gap_by_distribution: the in-dist ->
    # OOD drop per arm. The OOD-only breakdowns (answer type, domain, trap) are
    # the generalization claim; in-dist accuracy is the "did both arms learn it"
    # sanity check that the paper's logic requires.
    tables = {
        # Headline: balanced accuracy (base-rate-proof), in-dist vs OOD per arm.
        "balanced_by_distribution": metrics.balanced_accuracy(
            merged, ["model", "arm", "distribution"]
        ),
        "by_distribution": metrics.by_distribution(merged),  # raw accuracy (secondary)
        "by_seed_ood": metrics.by_seed(ood),
        "by_domain": metrics.by_domain(merged),
        "by_structure_ood": metrics.by_structure(ood),
        "by_structure_by_distribution": metrics.by_structure_distribution(merged),
        "by_perspective_ood": metrics.by_perspective(ood),
        "by_trap_ood": metrics.by_trap(ood),
        # Gap decomposition: total = length + reasoning. The reasoning gap
        # (why vs placebo) is the one the experiment actually cares about.
        "gap_total_by_distribution": metrics.gap(merged, "why", "demos", breakdown="distribution"),
        "gap_length_by_distribution": metrics.gap(
            merged, "placebo", "demos", breakdown="distribution"
        ),
        "gap_reasoning_by_distribution": metrics.gap(
            merged, "why", "placebo", breakdown="distribution"
        ),
        "gap_reasoning_ood_by_structure": metrics.gap(ood, "why", "placebo", breakdown="structure"),
        "gap_reasoning_ood_by_domain": metrics.gap(ood, "why", "placebo", breakdown="domain"),
    }
    for name, tbl in tables.items():
        tbl.to_parquet(paths["metrics"] / f"{name}.parquet", index=False)
        print(f"[metrics] {name} ({len(tbl)} rows)")
    return tables
