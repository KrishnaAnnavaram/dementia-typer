"""Command-line interface: `dementia-typer <command>`."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .config import Settings
from .data import SchemaError, build_table, load_tables
from .features import SET_DESCRIPTIONS
from .labels import CLASSES, unmapped
from .modeling import MODELS, SELECTORS


def cmd_synth(args, settings) -> int:
    from .synthetic import write

    for name, path in write(args.out, n_subjects=args.subjects, seed=args.seed).items():
        print(f"wrote {name}: {path}")
    return 0


def cmd_validate(args, settings) -> int:
    tables = load_tables(args.data)
    for name, frame in tables.items():
        print(f"{name}: {len(frame)} rows, schema OK")
    table = build_table(args.data, settings.join_window_days)
    texts = unmapped(table["dx1"])
    with_mri = int(table["IntraCranialVol"].notna().sum()) if "IntraCranialVol" in table else 0
    print(f"visits {len(table)}, participants {table['OASISID'].nunique()}, "
          f"visits with an MR session within {settings.join_window_days} days: {with_mri}")
    counts = table["group"].value_counts()
    print("groups: " + ", ".join(f"{c} {int(counts.get(c, 0))}" for c in CLASSES)
          + f", unmapped {int(table['group'].isna().sum())}")
    if texts:
        print("dx1 texts with no rule: " + "; ".join(texts[:20]))
        return 2
    return 0


def _print_experiment(exp) -> None:
    t, b = exp.test, exp.baseline_test
    ci = exp.ci.get("macro_f1", (float("nan"), float("nan")))
    print(f"set {exp.feature_set} ({SET_DESCRIPTIONS[exp.feature_set]}): chosen {exp.chosen['model']}"
          f"/{exp.chosen['selector']} (inner macro-F1 {exp.chosen['inner_macro_f1']:.3f})")
    print(f"  test macro-F1 {t['macro_f1']:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}), balanced accuracy "
          f"{t['balanced_accuracy']:.3f}, accuracy {t['accuracy']:.3f}, macro AUROC {t['macro_auroc_ovr']:.3f}, "
          f"ECE {t['ece']:.3f}")
    print("  recall: " + ", ".join(f"{k} {v:.2f}" for k, v in t["recall"].items())
          + (f" (no test visit: {', '.join(t['absent_classes'])})" if t["absent_classes"] else ""))
    print(f"  majority baseline: macro-F1 {b['macro_f1']:.3f}, accuracy {b['accuracy']:.3f}")


def cmd_train(args, settings) -> int:
    from .train import run_experiment, save

    table = build_table(args.data, settings.join_window_days)
    exp = run_experiment(table, args.feature_set, models=tuple(args.models), selectors=tuple(args.selectors),
                         k=args.k, seed=settings.seed, n_splits=settings.n_splits, nested=args.nested,
                         n_boot=args.bootstrap)
    _print_experiment(exp)
    if exp.nested:
        scores = [f["score"] for f in exp.nested]
        print(f"  nested CV macro-F1 (train participants): {sum(scores) / len(scores):.3f} over {len(scores)} folds")
    out = Path(args.out or settings.output_dir / f"set_{args.feature_set}")
    for name, path in save(exp, out).items():
        print(f"wrote {name}: {path}")
    return 0


def cmd_ablation(args, settings) -> int:
    from .train import run_experiment

    table = build_table(args.data, settings.join_window_days)
    for fs in ("A", "B", "C"):
        _print_experiment(run_experiment(table, fs, seed=settings.seed, n_splits=settings.n_splits,
                                         n_boot=args.bootstrap))
    print("Set C uses CDR. Its score is a leakage ceiling, not a valid estimate.")
    return 0


def cmd_leakage(args, settings) -> int:
    from .train import leakage_check

    r = leakage_check(build_table(args.data, settings.join_window_days), args.feature_set, settings.seed)
    print(f"forest macro-F1, visit-level split:       {r['visit_split']:.3f}")
    print(f"forest macro-F1, participant-level split: {r['participant_split']:.3f}")
    print(f"gap: {r['gap']:.3f}")
    return 0


def cmd_predict(args, settings) -> int:
    from .predict import predict_visit
    from .train import load

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    try:
        result = predict_visit(load(args.model), payload)
    except ValidationError as exc:
        print(f"error: the input does not match the schema:\n{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def cmd_demo(args, settings) -> int:
    from .synthetic import write
    from .train import leakage_check, run_experiment

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "synthetic"
        write(data, n_subjects=args.subjects, seed=args.seed)
        print(f"synthetic data: {args.subjects} participants (seed {args.seed})")
        cmd_validate(argparse.Namespace(data=data), settings)
        table = build_table(data, settings.join_window_days)
        r = leakage_check(table, "A", settings.seed)
        print(f"leak check (set A forest macro-F1): visit split {r['visit_split']:.3f}, participant split "
              f"{r['participant_split']:.3f}, gap {r['gap']:.3f}")
        for fs in ("A", "B", "C"):
            _print_experiment(run_experiment(table, fs, seed=settings.seed, n_splits=settings.n_splits,
                                             n_boot=args.bootstrap))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dementia-typer", description="Leakage-aware dementia group classification")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth", help="write synthetic clinical, FreeSurfer and demographics tables")
    s.add_argument("--out", type=Path, default=Path("data/synthetic"))
    s.add_argument("--subjects", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    def with_data(name, func, help_text):
        q = sub.add_parser(name, help=help_text)
        q.add_argument("--data", type=Path, required=True, help="folder with clinical.csv (+ freesurfer, demographics)")
        q.set_defaults(func=func)
        return q

    with_data("validate", cmd_validate, "check the tables, the visit-to-MR join and the label rules")
    q = with_data("train", cmd_train, "participant hold-out, grouped model selection, test, model card")
    q.add_argument("--feature-set", choices=["A", "B", "C"], default="A")
    q.add_argument("--models", nargs="+", choices=[m for m in MODELS if m != "majority"],
                   default=["logreg", "forest"])
    q.add_argument("--selectors", nargs="+", choices=list(SELECTORS), default=["none", "mi"])
    q.add_argument("--k", type=int, default=8, help="features kept by a selector")
    q.add_argument("--nested", action="store_true", help="also run the outer grouped CV")
    q.add_argument("--bootstrap", type=int, default=200)
    q.add_argument("--out", type=Path)
    q = with_data("ablation", cmd_ablation, "compare feature sets A, B and C")
    q.add_argument("--bootstrap", type=int, default=200)
    q = with_data("leakage", cmd_leakage, "visit-level vs. participant-level split")
    q.add_argument("--feature-set", choices=["A", "B", "C"], default="A")
    r = sub.add_parser("predict", help="predict one visit from a JSON file")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--input", type=Path, required=True)
    r.set_defaults(func=cmd_predict)
    d = sub.add_parser("demo", help="run the full path on synthetic data")
    d.add_argument("--subjects", type=int, default=300)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--bootstrap", type=int, default=200)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    try:
        return args.func(args, settings)
    except (SchemaError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
