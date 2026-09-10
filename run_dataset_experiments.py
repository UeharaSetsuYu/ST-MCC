"""Run the fixed protocol on every bundled dataset and aggregate mean +/- std."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from config import Config, OUTPUT_ROOT
from data_load import DATASET_LOADERS, load_dataset
from train import train


DEFAULT_SEEDS = (61, 62, 63, 64, 65)
METRICS = ("ACC", "NMI", "ARI", "PUR")


def _json_payload(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "state_dict"}


def run_one(dataset_path: Path, output: Path, seed: int) -> dict:
    run_dir = output / dataset_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / f"seed_{seed}.json"
    log_path = run_dir / f"seed_{seed}.log"

    config = Config(data=str(dataset_path), output=str(run_dir), seed=seed)
    started = time.time()
    captured = io.StringIO()
    with redirect_stdout(captured):
        result = train(config)
    result["runtime_seconds"] = time.time() - started
    payload = _json_payload(result)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log_path.write_text(captured.getvalue(), encoding="utf-8")
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def aggregate(dataset_path: Path, runs: list[dict]) -> dict:
    data = load_dataset(str(dataset_path), Config.missing_rate, DEFAULT_SEEDS[0])
    summary = {
        "dataset": data.dataset_name,
        "file": dataset_path.name,
        "samples": len(data.labels),
        "classes": data.num_clusters,
        "source_views": data.source_view_count,
        "used_views": [index + 1 for index in data.selected_view_indices],
        "input_dims": data.input_dims,
        "runs": len(runs),
        "runtime_seconds": float(sum(run["runtime_seconds"] for run in runs)),
        "metrics": {},
    }
    for metric in METRICS:
        values = np.asarray([run["final"][metric] for run in runs], dtype=np.float64)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    return summary


def format_metric(metric: dict) -> str:
    return f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def write_outputs(output: Path, summaries: list[dict], seeds: tuple[int, ...]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "seeds": list(seeds),
            "repetitions": len(seeds),
            "std": "sample standard deviation (ddof=1)",
            "missing_rate": Config.missing_rate,
            "pretrain_epochs": Config.pretrain_epochs,
            "joint_epochs": Config.joint_epochs,
            "batch_size": Config.batch_size,
            "learning_rate": Config.learning_rate,
            "views": "first two source views for every dataset",
            "hyperparameter_tuning": "none",
        },
        "datasets": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Dataset",
                "Samples",
                "Classes",
                "SourceViews",
                "UsedViews",
                "InputDims",
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
        for item in summaries:
            row = [
                item["dataset"],
                item["samples"],
                item["classes"],
                item["source_views"],
                ",".join(map(str, item["used_views"])),
                ",".join(map(str, item["input_dims"])),
            ]
            for metric in METRICS:
                row.extend(
                    [item["metrics"][metric]["mean"], item["metrics"][metric]["std"]]
                )
            row.append(item["runtime_seconds"])
            writer.writerow(row)

    best = {
        metric: max(summaries, key=lambda item: item["metrics"][metric]["mean"])["dataset"]
        for metric in METRICS
    }
    lines = [
        "# ST-MCC 数据集筛选实验报告",
        "",
        "## 1. 实验目的",
        "",
        "在不针对数据集调整模型超参数的条件下，评估当前两视图ST-MCC训练流程在项目内置数据集上的稳定性与适用性。",
        "",
        "## 2. 实验协议",
        "",
        f"- 重复次数：{len(seeds)}次，随机种子为 `{', '.join(map(str, seeds))}`。",
        f"- 缺失率：{Config.missing_rate:.1f}；每个不完整样本随机保留一个视图。",
        f"- 训练：{Config.pretrain_epochs}轮重构预训练 + {Config.joint_epochs}轮联合训练。",
        f"- batch size：{Config.batch_size}；学习率：{Config.learning_rate:g}。",
        "- 所有网络结构、损失权重、教师参数和优化器设置均保持默认，无数据集级调参。",
        "- 当前模型仅支持两视图，因此所有多视图数据集固定使用源文件中的前两个视图。",
        "- Flower17提供距离矩阵；加载器使用每个距离视图的正值中位数作为固定尺度，将其转换为RBF相似度特征。",
        "- 簇数K由数据集标签中的类别数给出；标签不参与教师构造、网络训练或模型选择。",
        "- 表中标准差为5次运行的样本标准差（ddof=1）。",
        "",
        "## 3. 数据集概况",
        "",
        "| 数据集 | 样本数 | 类别数 | 原始视图数 | 使用视图 | 输入维度 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['dataset']} | {item['samples']} | {item['classes']} | "
            f"{item['source_views']} | {','.join(map(str, item['used_views']))} | "
            f"{','.join(map(str, item['input_dims']))} |"
        )
    lines.extend(
        [
            "",
            "## 4. 聚类结果",
            "",
            "| 数据集 | ACC ↑ | NMI ↑ | ARI ↑ | PUR ↑ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        lines.append(
            "| "
            + item["dataset"]
            + " | "
            + " | ".join(format_metric(item["metrics"][metric]) for metric in METRICS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 5. 结果解读",
            "",
            f"在本次固定协议下，平均ACC最高的数据集是{best['ACC']}，平均NMI最高的是{best['NMI']}，平均ARI最高的是{best['ARI']}，平均PUR最高的是{best['PUR']}。这些结论只反映当前两视图、50%缺失率和默认参数设置下的绝对表现，不构成与其他方法的优劣比较。",
            "",
            "跨数据集结果可以用于初步筛选更适合当前模型的数据，但不能把不同类别数、样本数和特征类型的数据集指标直接解释为模型泛化能力的严格排序。尤其是多视图数据集只使用前两个视图，Flower17还包含距离到相似度的输入转换。",
            "",
            "## 6. 限制与风险",
            "",
            "1. 当前网络是两视图架构，未使用100Leaves、ALOI_100、Flower17、HW和Scene_15的其余视图。",
            "2. 结构教师使用自适应图邻域、双向缺失细化与软置信伪标签，其跨数据集贡献尚需消融验证。",
            "3. 本实验没有基线方法、模块消融或缺失率敏感性实验，因此只能回答“默认模型在哪些数据集上更稳定”，不能支持SOTA或模块有效性结论。",
            "4. 随机种子同时控制网络初始化和缺失掩码，因此标准差反映二者共同造成的波动。",
            "5. 本次运行中ALOI_100在多个seed出现kNN图不完全连通警告，固定图教师可能无法可靠表达其全局结构；这是相关诊断而非已证实的因果解释。",
            "6. Windows MKL在DHA的KMeans阶段报告线程相关内存泄漏警告；所有运行均正常结束，但扩大重复次数时宜采用进程隔离。",
            "",
            "## 7. Claim–Evidence核对",
            "",
            "- Claim：报告反映无数据集级调参的默认性能。Evidence：所有运行共享同一Config，仅数据路径、随机种子和由数据集决定的K不同。Status：supported。",
            "- Claim：某数据集更适合当前模型。Evidence：只能由本表的均值与标准差进行初步判断。Status：limited；仍需基线和多缺失率实验。",
            "- Claim：当前方法优于已有方法或达到SOTA。Evidence：本实验未运行外部基线。Status：unsupported，本报告不作此声明。",
            "",
            "## 8. 审稿式自检",
            "",
            "- 贡献：本报告提供跨数据集默认协议筛选证据，但不构成新方法贡献。",
            "- 写作清晰度：数据选择、预处理、种子和统计方式均已明确。",
            "- 实验强度：有5次重复，但缺少强基线与显著性检验。",
            "- 评估完整性：缺少模块消融、不同缺失率和全视图版本。",
            "- 方法合理性：两视图限制与BDGP特定教师修正是主要外部有效性风险。",
            "",
        ]
    )
    (output / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--datasets", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output = OUTPUT_ROOT
    seeds = tuple(args.seeds)
    available = {
        path.stem.lower(): path
        for path in dataset_dir.glob("*.mat")
        if path.stem.lower() in DATASET_LOADERS
    }
    requested = [name.lower() for name in args.datasets] if args.datasets else sorted(available)
    missing = [name for name in requested if name not in available]
    if missing:
        raise FileNotFoundError(f"Datasets not found or unsupported: {missing}")

    summaries = []
    for name in requested:
        dataset_path = available[name]
        runs = []
        for seed in seeds:
            print(f"START dataset={dataset_path.stem} seed={seed}", flush=True)
            run = run_one(dataset_path, output, seed)
            runs.append(run)
            print(
                f"DONE dataset={dataset_path.stem} seed={seed} "
                + " ".join(f"{metric}={run['final'][metric]:.4f}" for metric in METRICS),
                flush=True,
            )
        summary = aggregate(dataset_path, runs)
        summaries.append(summary)
        print(
            f"SUMMARY dataset={summary['dataset']} "
            + " ".join(
                f"{metric}={format_metric(summary['metrics'][metric])}" for metric in METRICS
            ),
            flush=True,
        )
        write_outputs(output, summaries, seeds)

    print(f"REPORT {output / 'experiment_report.md'}", flush=True)


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("Times: ", time.time() - start_time)
