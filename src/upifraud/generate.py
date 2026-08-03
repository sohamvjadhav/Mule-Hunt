"""Synthetic UPI-style transaction graph generation."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

FRAUD_AMOUNT = 9999.0


def run_gen_fraud_graph(
    output_dir: Path,
    scale: float = 0.001,
    num_fraud_rings: int | None = None,
    workers: int = 2,
    hardness: str = "low",
) -> None:
    try:
        from gen_fraud_graph import Config, FraudGraphGenerator
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "the synthetic generator is a git dependency and is not included "
            "with the PyPI package; install it with:\n"
            '  pip install "gen-fraud-graph @ git+https://github.com/SantanderAI/gen-fraud-graph.git"'
        ) from e

    config = Config(
        scale_factor=scale,
        num_fraud_rings=num_fraud_rings,
        embedding_provider="fake",
        workers=workers,
        output_dir=str(output_dir),
        hardness=hardness,
    )
    FraudGraphGenerator(config).run()


def generate_toy(
    output_dir: Path,
    n_accounts: int = 300,
    n_tx: int = 2500,
    n_rings: int = 5,
    ring_min: int = 4,
    ring_max: int = 7,
    seed: int = 42,
) -> None:
    """Write a small graph in the same CSV schema as gen-fraud-graph."""
    rng = random.Random(seed)
    accounts_dir = output_dir / "accounts"
    tx_dir = output_dir / "transactions"
    fraud_dir = output_dir / "fraud"
    accounts_dir.mkdir(parents=True, exist_ok=True)
    tx_dir.mkdir(parents=True, exist_ok=True)
    fraud_dir.mkdir(parents=True, exist_ok=True)

    account_ids = [f"acc_{i}" for i in range(n_accounts)]
    accounts = pd.DataFrame(
        {
            "account_id": account_ids,
            "customer_name": [f"Customer {i}" for i in range(n_accounts)],
            "balance": [rng.uniform(100.0, 100_000.0) for _ in range(n_accounts)],
            "risk_score": [rng.uniform(0.0, 1.0) for _ in range(n_accounts)],
            "creation_date": [
                f"2023-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}" for _ in range(n_accounts)
            ],
        }
    )
    accounts.to_csv(accounts_dir / "accounts_0_0.csv", index=False)

    rows = []
    for i in range(n_tx):
        src, dst = rng.sample(account_ids, 2)
        rows.append(
            {
                "tx_id": f"tx_{i}",
                "src_id": src,
                "dst_id": dst,
                "amount": rng.uniform(10.0, 500.0),
                "timestamp": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
                "description": "UPI transfer",
            }
        )

    ring_accounts: list[list[str]] = []
    fraud_tx_rows: list[dict] = []
    used = set()
    next_tx = n_tx
    for ring_idx in range(n_rings):
        size = rng.randint(ring_min, ring_max)
        pool = [a for a in account_ids if a not in used]
        if len(pool) < size:
            break
        members = rng.sample(pool, size)
        used.update(members)
        ring_accounts.append(members)
        for i in range(size):
            src = members[i]
            dst = members[(i + 1) % size]
            fraud_tx_rows.append(
                {
                    "tx_id": f"tx_{next_tx}",
                    "src_id": src,
                    "dst_id": dst,
                    "amount": FRAUD_AMOUNT,
                    "timestamp": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
                    "description": "SUSPICIOUS CYCLE",
                }
            )
            next_tx += 1
        if ring_idx % 2 == 0:
            for i in range(2):
                src = members[i]
                dst = members[(i + 2) % size]
                fraud_tx_rows.append(
                    {
                        "tx_id": f"tx_{next_tx}",
                        "src_id": src,
                        "dst_id": dst,
                        "amount": FRAUD_AMOUNT,
                        "timestamp": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
                        "description": "SUSPICIOUS SPLIT",
                    }
                )
                next_tx += 1
        for _ in range(size):
            src = rng.choice(members)
            dst = rng.choice([a for a in account_ids if a not in members])
            rows.append(
                {
                    "tx_id": f"tx_{next_tx}",
                    "src_id": src,
                    "dst_id": dst,
                    "amount": rng.uniform(10.0, 500.0),
                    "timestamp": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
                    "description": "UPI transfer",
                }
            )
            next_tx += 1

    pd.DataFrame(rows).to_csv(tx_dir / "transactions_0_0.csv", index=False)
    pd.DataFrame(fraud_tx_rows).to_csv(fraud_dir / "transactions_fraud.csv", index=False)

    cases = [
        {
            "pattern_id": f"pat_{i}",
            "start_acc_id": members[0],
            "pattern_type": "cycle",
            "depth": len(members),
            "involved_accounts": "|".join(members),
        }
        for i, members in enumerate(ring_accounts)
    ]
    pd.DataFrame(cases).to_csv(fraud_dir / "fraud_cases.csv", index=False)
