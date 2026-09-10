"""Run repeated experiments across fixed missing rates and report mean +/- std."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from config import Config, OUTPUT_ROOT
from data_load import load_dataset
from train import train


MISSING_RATES = (0.1, 0.3, 0.5, 0.7)
DEFAULT_SEEDS = (61, 62, 63, 64, 65)
METRICS = ("ACC", "NMI", "ARI", "PUR")


def _safe_name(value: str) -> str:
    return Path(value).stem.lower().replace("-", "_").replace(" ", "_")


def _rate_name(missing_rate: float) -> str:
    return f"mr_{missing_rate:.1f}".replace(".", "p")


def run_one(
    dataname: str,
    dataset_name: str,
    missing_rate: float,
    seed: int,
) -> dict:
    run_dir = OUTPUT_ROOT / _safe_name(dataset_name) / _rate_name(missing_rate)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / f"seed_{seed}.json"
    log_path = run_dir / f"seed_{seed}.log"

    config = Config(
        data=dataname,
        output=str(run_dir),
        seed=seed,
        missing_rate=missing_rate,
    )
    captured = io.StringIO()
    started = time.time()
    with warnings.catch_warnings(), redirect_stdout(captured), redirect_stderr(captured):
        warnings.simplefilter("always")
        result = train(config)
    result["runtime_seconds"] = time.time() - started
    payload = {key: value for key, value in result.items() if key != "state_dict"}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log_path.write_text(captured.getvalue(), encoding="utf-8")
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def aggregate(data, missing_rate: float, seeds: tuple[int, ...], runs: list[dict]) -> dict:
    summary = {
        "dataset": data.dataset_name,
        "source_path": data.source_path,
        "missing_rate": missing_rate,
        "samples": len(data.labels),
        "classes": data.num_clusters,
        "input_dims": data.input_dims,
        "batch_size": data.batch_size,
        "seeds": list(seeds),
        "runs": len(runs),
        "runtime_seconds": float(sum(run["runtime_seconds"] for run in runs)),
        "metrics": {},
    }
    for metric in METRICS:
        values = np.asarray([run["final"][metric] for run in runs], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite {metric} result for {data.dataset_name}, missing_rate={missing_rate}"
            )
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return summary


def format_metric(metric: dict) -> str:
    return f"{metric['mean']:.4f} +/- {metric['std']:.4f}"


def write_summaries(summaries: list[dict], seeds: tuple[int, ...]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "missing_rates": list(MISSING_RATES),
            "seeds": list(seeds),
            "repetitions": len(seeds),
            "std": "sample standard deviation (ddof=1)",
            "overwrite": "same dataset, missing rate, and seed are retrained and overwritten",
        },
        "results": summaries,
    }
    (OUTPUT_ROOT / "missing_rate_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (OUTPUT_ROOT / "missing_rate_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Dataset",
                "MissingRate",
                "Samples",
                "Classes",
                "InputDims",
                "BatchSize",
                "Seeds",
                "ACC_mean",
                "ACC_std",
                "NMI_mean",
                "NMI_std",
                "ARI_mean",
                "ARI_std",
                "PUR_mean",
                "PUR_std",
                "RuntimeSeconds",
            ]
        )
        for summary in summaries:
            row = [
                summary["dataset"],
                summary["missing_rate"],
                summary["samples"],
                summary["classes"],
                ",".join(map(str, summary["input_dims"])),
                summary["batch_size"],
                ",".join(map(str, summary["seeds"])),
            ]
            for metric in METRICS:
                row.extend(
                    [
                        summary["metrics"][metric]["mean"],
                        summary["metrics"][metric]["std"],
                    ]
                )
            row.append(summary["runtime_seconds"])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    summaries: list[dict] = []

    for dataname in args.datasets:
        for missing_rate in MISSING_RATES:
            data = load_dataset(dataname, missing_rate, seeds[0])
            runs = []
            for seed in seeds:
                print(
                    f"START dataset={data.dataset_name} batch_size={data.batch_size} "
                    f"seed={seed} missing_rate={missing_rate:.1f}",
                    flush=True,
                )
                run = run_one(dataname, data.dataset_name, missing_rate, seed)
                runs.append(run)
                print(
                    f"DONE dataset={data.dataset_name} missing_rate={missing_rate:.1f} "
                    f"seed={seed} "
                    + " ".join(
                        f"{metric}={run['final'][metric]:.4f}" for metric in METRICS
                    ),
                    flush=True,
                )

            summary = aggregate(data, missing_rate, seeds, runs)
            summaries.append(summary)
            print(
                f"SUMMARY dataset={data.dataset_name} missing_rate={missing_rate:.1f} "
                + " ".join(
                    f"{metric}={format_metric(summary['metrics'][metric])}"
                    for metric in METRICS
                ),
                flush=True,
            )
            write_summaries(summaries, seeds)

    print(f"RESULTS {OUTPUT_ROOT / 'missing_rate_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
