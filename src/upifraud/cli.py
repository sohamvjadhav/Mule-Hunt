"""Command-line interface: generate / train-gnn / train-baseline / evaluate / serve / demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from . import __version__
from .api import create_app
from .baseline import evaluate_baseline, train_baseline
from .dataset import load_graph
from .evaluate import ring_recovery, top_fraud_accounts
from .generate import generate_toy, run_gen_fraud_graph
from .train import evaluate_gnn, train_gnn


def _print_table(rows: list[dict], cols: list[str]) -> None:
    widths = [max(len(col), *(len(str(r[col])) for r in rows)) for col in cols]
    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))


def cmd_generate(args) -> None:
    out = Path(args.output)
    if args.toy:
        generate_toy(out, n_accounts=args.toy_accounts, n_tx=args.toy_tx, n_rings=args.rings, seed=args.seed)
    else:
        run_gen_fraud_graph(
            out,
            scale=args.scale,
            num_fraud_rings=args.rings,
            workers=args.workers,
            hardness=args.hardness,
        )
    print(f"generated graph CSVs in {out.resolve()}")


def cmd_train_gnn(args) -> None:
    data = load_graph(
        Path(args.data),
        with_amount_stats=args.amount_stats,
        split=args.split,
        test_rings=args.test_rings,
        seed=args.seed,
    )
    out = Path(args.out_dir)
    result = train_gnn(
        data,
        model_name=args.model,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        out_dir=out,
    )
    eval_ = evaluate_gnn(data, result["scores"], split="test")
    rec = ring_recovery(data, result["scores"], split="test")
    print(json.dumps({"model": args.model, "test": eval_, "ring_recovery": rec}, indent=2))
    torch.save(data, out / "graph.pt")


def cmd_train_baseline(args) -> None:
    data = load_graph(
        Path(args.data),
        with_amount_stats=args.amount_stats,
        split=args.split,
        test_rings=args.test_rings,
        seed=args.seed,
    )
    result = train_baseline(data, model_name=args.model, seed=args.seed, out_dir=Path(args.out_dir))
    eval_ = evaluate_baseline(data, result["scores"], split="test")
    rec = ring_recovery(data, result["scores"], split="test")
    print(json.dumps({"model": args.model, "test": eval_, "ring_recovery": rec}, indent=2))


def cmd_evaluate(args) -> None:
    out = Path(args.out_dir)
    data = torch.load(out / "graph.pt", map_location="cpu", weights_only=False)
    rows = []
    present = sorted(
        [p.stem for p in out.glob("*.joblib")] + [p.stem for p in out.glob("*.pt") if p.stem != "graph"]
    )
    for name in present:
        if name in ("rf", "hgb", "xgb"):
            path = out / f"{name}.joblib"
            if not path.exists():
                continue
            import joblib

            from .train import standardize

            bargs = json.loads((out / f"{name}_args.json").read_text())
            xb, _, _ = standardize(data.x, torch.tensor(bargs["mean"]), torch.tensor(bargs["std"]))
            scores = joblib.load(path).predict_proba(xb.numpy())[:, 1]
        else:
            ckpt = out / f"{name}.pt"
            if not ckpt.exists():
                continue
            args_ = json.loads((out / f"{name}_args.json").read_text())
            from .models import build_model
            from .train import standardize

            model = build_model(name, int(args_["in_dim"]), int(args_["hidden"]))
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.eval()
            x, _, _ = standardize(data.x, torch.tensor(args_["mean"]), torch.tensor(args_["std"]))
            with torch.no_grad():
                scores = model(x, data.edge_index).sigmoid().numpy()
        eval_ = evaluate_gnn(data, scores, split="test") if name in ("gcn", "sage") else evaluate_baseline(data, scores, split="test")
        rec = ring_recovery(data, scores, split="test")
        rows.append(
            {
                "model": name,
                "auc": round(eval_["auc"], 4),
                "ap": round(eval_["ap"], 4),
                "mean_ring_recall": round(rec["mean_ring_recall"], 4),
                "fraud_hit@k": round(rec["fraud_hit_rate_at_k"], 4),
            }
        )
    _print_table(rows, ["model", "auc", "ap", "mean_ring_recall", "fraud_hit@k"])


def cmd_serve(args) -> None:
    import uvicorn


    frontend = Path(args.frontend) if args.frontend else None
    app = create_app(Path(args.out_dir), Path(args.out_dir) / "graph.pt", frontend_dir=frontend)
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_demo(args) -> None:
    data_dir = Path(args.data)
    out = Path(args.out_dir)
    if not (data_dir / "accounts").exists():
        cmd_generate(
            argparse.Namespace(
                output=str(data_dir), toy=args.toy, scale=args.scale, rings=args.rings,
                workers=args.workers, seed=args.seed, toy_accounts=args.toy_accounts, toy_tx=args.toy_tx,
                hardness=args.hardness,
            )
        )
    print("== training GNN ==")
    cmd_train_gnn(argparse.Namespace(
        data=str(data_dir), out_dir=str(out), model=args.model, hidden=args.hidden,
        epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed,
        amount_stats=args.amount_stats, split="rings", test_rings=args.test_rings,
    ))
    print("== training baselines ==")
    for base in ("rf", "hgb"):
        cmd_train_baseline(argparse.Namespace(
            data=str(data_dir), out_dir=str(out), model=base, seed=args.seed,
            amount_stats=args.amount_stats, split="rings", test_rings=args.test_rings,
        ))
    print("== comparison ==")
    cmd_evaluate(argparse.Namespace(out_dir=str(out)))
    print("== top predicted fraud accounts ==")
    data = torch.load(out / "graph.pt", map_location="cpu", weights_only=False)
    args_ = json.loads((out / f"{args.model}_args.json").read_text())
    from .models import build_model
    from .train import standardize

    model = build_model(args.model, int(args_["in_dim"]), int(args_["hidden"]))
    model.load_state_dict(torch.load(out / f"{args.model}.pt", map_location="cpu"))
    model.eval()
    x, _, _ = standardize(data.x, torch.tensor(args_["mean"]), torch.tensor(args_["std"]))
    with torch.no_grad():
        scores = model(x, data.edge_index).sigmoid().numpy()
    for row in top_fraud_accounts(data, scores, k=15):
        print(f"  #{row['rank']:>2} {row['account_id']:<12} score={row['risk_score']:.3f} "
              f"ring={row['ring_id']:<3} label={row['true_label']}")


def cmd_benchmark(args) -> None:
    """Run the hardness matrix: GNN + baselines at low/medium/high difficulty."""
    import shutil

    from .evaluate import ring_recovery

    root = Path(args.root)
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    for hardness in args.hardness:
        data_dir = root / f"data_{hardness}"
        out = root / f"models_{hardness}"
        if (not (data_dir / "accounts").exists()) or args.regenerate:
            shutil.rmtree(data_dir, ignore_errors=True)
            cmd_generate(
                argparse.Namespace(
                    output=str(data_dir), toy=False, scale=args.scale, rings=args.rings,
                    workers=1, seed=args.seed, toy_accounts=0, toy_tx=0, hardness=hardness,
                )
            )
        print(f"== training {hardness} ==")
        data = load_graph(
            data_dir,
            with_amount_stats=args.amount_stats,
            split="rings",
            test_rings=args.test_rings,
            seed=args.seed,
        )
        gnn = train_gnn(
            data, model_name=args.model, hidden=args.hidden, epochs=args.epochs,
            lr=args.lr, patience=args.patience, seed=args.seed, out_dir=out,
        )
        rows = []
        gnn_test = evaluate_gnn(data, gnn["scores"], split="test")
        gnn_ring = ring_recovery(data, gnn["scores"], split="test")
        rows.append({
            "hardness": hardness, "model": args.model,
            "auc": gnn_test["auc"], "ap": gnn_test["ap"],
            "mean_ring_recall": gnn_ring["mean_ring_recall"],
        })
        torch.save(data, out / "graph.pt")
        for base in args.baselines:
            result = train_baseline(data, model_name=base, seed=args.seed, out_dir=out)
            eval_ = evaluate_baseline(data, result["scores"], split="test")
            rec = ring_recovery(data, result["scores"], split="test")
            rows.append({
                "hardness": hardness, "model": base,
                "auc": eval_["auc"], "ap": eval_["ap"],
                "mean_ring_recall": rec["mean_ring_recall"],
            })
        for r in rows:
            print(f"  {r['hardness']:<7} {r['model']:<4} auc={r['auc']:.3f} "
                  f"ap={r['ap']:.3f} ring_recall={r['mean_ring_recall']:.3f}")
        outcomes[hardness] = rows
    (results_dir / "benchmark.json").write_text(
        json.dumps(outcomes, indent=2, sort_keys=True)
    )
    print(f"benchmark results written to {results_dir / 'benchmark.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="upifraud", description="UPI fraud detection with GNNs")
    parser.add_argument("--version", action="version", version=f"upi-fraud-gnn {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="generate synthetic transaction graph")
    p.add_argument("--output", default="data/raw")
    p.add_argument("--scale", type=float, default=0.001)
    p.add_argument("--rings", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--hardness", choices=["low", "medium", "high"], default="low")
    p.add_argument("--toy", action="store_true", help="use the tiny built-in generator (tests)")
    p.add_argument("--toy-accounts", type=int, default=300)
    p.add_argument("--toy-tx", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train-gnn", help="train GCN/GraphSAGE")
    p.add_argument("--data", default="data/raw")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--model", choices=["gcn", "sage"], default="sage")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--split", choices=["rings", "random"], default="rings")
    p.add_argument("--test-rings", type=int, default=3)
    p.add_argument("--amount-stats", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train_gnn)

    p = sub.add_parser("train-baseline", help="train RF/XGBoost baseline")
    p.add_argument("--data", default="data/raw")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--model", choices=["rf", "hgb", "xgb"], default="rf")
    p.add_argument("--split", choices=["rings", "random"], default="rings")
    p.add_argument("--test-rings", type=int, default=3)
    p.add_argument("--amount-stats", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train_baseline)

    p = sub.add_parser("evaluate", help="compare all saved models")
    p.add_argument("--out-dir", default="models")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("serve", help="run the FastAPI risk-scoring service + dashboard")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--frontend", default=None, help="path to frontend dir (default: repo frontend/)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("demo", help="end-to-end run on a small graph")
    p.add_argument("--data", default="data/raw")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--scale", type=float, default=0.0001)
    p.add_argument("--rings", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--toy-accounts", type=int, default=300)
    p.add_argument("--toy-tx", type=int, default=2500)
    p.add_argument("--model", choices=["gcn", "sage"], default="sage")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--test-rings", type=int, default=3)
    p.add_argument("--amount-stats", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("benchmark", help="run the hardness benchmark matrix (low/medium/high)")
    p.add_argument("--root", default="bench")
    p.add_argument("--scale", type=float, default=0.001)
    p.add_argument("--rings", type=int, default=None)
    p.add_argument("--hardness", nargs="+", choices=["low", "medium", "high"], default=["low", "medium", "high"])
    p.add_argument("--model", choices=["gcn", "sage"], default="sage")
    p.add_argument("--baselines", nargs="+", choices=["rf", "hgb", "xgb"], default=["rf"])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--test-rings", type=int, default=3)
    p.add_argument("--amount-stats", action="store_true")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
