"""Run the fixed ST-MCC protocol on registered datasets and aggregate results."""

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
from data_load import DATASET_PATHS, load_dataset
from train import train


DEFAULT_SEEDS = (61, 62, 63, 64, 65)
METRICS = ("ACC", "NMI", "ARI", "PUR")
REQUESTED_DATASETS = (
    "COIL20",
    "Caltech101-20",
    "CiteSeer",
    "CUB",
    "Fashion",
    "Reuters_dim10",
    "LandUse_21",
    "NUSWIDE",
    "LGG",
    "MNIST_USPS",
    "NGs",
    "RGB_D",
)


def _safe_name(dataname: str) -> str:
    return dataname.lower().replace("-", "_")


def run_one(dataname: str, output: Path, seed: int) -> dict:
    run_dir = output / _safe_name(dataname)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / f"seed_{seed}.json"
    log_path = run_dir / f"seed_{seed}.log"

    config = Config(data=dataname, output=str(run_dir), seed=seed)
    started = time.time()
    captured = io.StringIO()
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


def aggregate(dataname: str, runs: list[dict], output: Path) -> dict:
    data = load_dataset(dataname, Config.missing_rate, DEFAULT_SEEDS[0])
    log_files = sorted((output / _safe_name(dataname)).glob("seed_*.log"))
    warning_lines = []
    for log_file in log_files:
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Warning:" in line and line not in warning_lines:
                warning_lines.append(line.strip())
    item = {
        "dataset": data.dataset_name,
        "source_path": data.source_path,
        "samples": len(data.labels),
        "classes": data.num_clusters,
        "source_views": data.source_view_count,
        "used_views": [index + 1 for index in data.selected_view_indices],
        "input_dims": data.input_dims,
        "batch_size": data.batch_size,
        "runs": len(runs),
        "runtime_seconds": float(sum(run["runtime_seconds"] for run in runs)),
        "warnings": warning_lines,
        "metrics": {},
    }
    for metric in METRICS:
        values = np.asarray([run["final"][metric] for run in runs], dtype=np.float64)
        item["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return item


def format_metric(metric: dict) -> str:
    return f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def write_outputs(
    output: Path,
    summaries: list[dict],
    incompatible: list[dict],
    seeds: tuple[int, ...],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "seeds": list(seeds),
            "repetitions": len(seeds),
            "std": "sample standard deviation (ddof=1)",
            "missing_rate": Config.missing_rate,
            "pretrain_epochs": Config.pretrain_epochs,
            "joint_epochs": Config.joint_epochs,
            "learning_rate": Config.learning_rate,
            "batch_size_rule": "64 if n<500; 256 if n<5000; otherwise 1024",
            "views": "first two source views for every compatible dataset",
            "hyperparameter_tuning": "none",
        },
        "datasets": summaries,
        "incompatible": incompatible,
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
                "BatchSize",
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
                item["batch_size"],
            ]
            for metric in METRICS:
                row.extend([item["metrics"][metric]["mean"], item["metrics"][metric]["std"]])
            row.append(item["runtime_seconds"])
            writer.writerow(row)

    best = (
        {
            metric: max(summaries, key=lambda item: item["metrics"][metric]["mean"])["dataset"]
            for metric in METRICS
        }
        if summaries
        else {}
    )
    lines = [
        "# ST-MCC 扩展数据集默认配置实验报告",
        "",
        "## 1. 实验目的",
        "",
        "本实验在不进行数据集级模型调参的条件下，评估当前两视图ST-MCC在新增多视图数据集上的聚类表现与训练稳定性。",
        "",
        "## 2. 实验协议",
        "",
        f"- 每个兼容数据集重复{len(seeds)}次，seed为 `{', '.join(map(str, seeds))}`。",
        f"- 缺失率固定为{Config.missing_rate:.1f}，每个不完整样本随机保留一个视图。",
        f"- 训练固定为{Config.pretrain_epochs}轮重构预训练和{Config.joint_epochs}轮联合训练，学习率为{Config.learning_rate:g}。",
        "- batch size严格按样本数确定：n<500为64，500≤n<5000为256，n≥5000为1024。",
        "- 模型、教师、损失权重和优化器的其他设置均保持不变。",
        "- 当前模型严格支持两个视图；多视图文件固定选择前两个真实视图，不合成额外视图。",
        "- 已知簇数K从数据标签的类别数读取；标签不进入教师、损失或模型选择。",
        "- 结果报告均值±样本标准差（ddof=1）。",
        "",
        "## 3. 数据集与实际配置",
        "",
        "| 数据集 | 样本 | 类别 | 原始视图 | 使用视图 | 输入维度 | Batch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['dataset']} | {item['samples']} | {item['classes']} | "
            f"{item['source_views']} | {','.join(map(str, item['used_views']))} | "
            f"{','.join(map(str, item['input_dims']))} | {item['batch_size']} |"
        )
    for item in incompatible:
        lines.append(
            f"| {item['dataset']} | {item.get('samples', '-')} | {item.get('classes', '-')} | "
            f"{item.get('source_views', '-')} | 不兼容 | - | {item.get('batch_size', '-')} |"
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
            "| " + item["dataset"] + " | "
            + " | ".join(format_metric(item["metrics"][metric]) for metric in METRICS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 5. 结果解读",
            "",
            (
                f"在本次固定协议下，平均ACC最高的数据集为{best['ACC']}，平均NMI最高的数据集为{best['NMI']}，平均ARI最高的数据集为{best['ARI']}，平均PUR最高的数据集为{best['PUR']}。该排序只描述当前默认协议下的绝对指标，不构成与外部方法的性能比较。"
                if best
                else "当前尚无兼容数据集的完整结果。"
            ),
            "",
            "不同数据集的样本规模、类别数、特征类型和原始视图数差异明显，因此不能把跨数据集指标差异直接归因于模型本身。均值用于描述典型表现，标准差用于识别对初始化和缺失掩码敏感的数据集。",
            "",
            "## 6. 不兼容数据与限制",
            "",
        ]
    )
    if incompatible:
        for item in incompatible:
            lines.append(f"- **{item['dataset']}**：{item['reason']}")
    else:
        lines.append("- 所有请求数据集均完成训练。")
    lines.extend(
        [
            "- 多视图数据仅使用前两个视图，因此结论不代表利用全部视图时的模型性能。",
            "- 结构教师使用自适应图邻域、双向缺失细化与软置信伪标签；其贡献仍需消融验证。",
            "- 本实验未包含外部基线、模块消融或不同缺失率，因此不能支持SOTA或模块因果有效性结论。",
            "- seed同时控制模型初始化和缺失掩码，标准差反映两类随机性的共同影响。",
            "",
            "## 7. 运行警告",
            "",
        ]
    )
    warning_items = [(item["dataset"], item["warnings"]) for item in summaries if item["warnings"]]
    if warning_items:
        for dataset, messages in warning_items:
            lines.append(f"- {dataset}：" + "；".join(messages))
    else:
        lines.append("- 未记录到影响完成状态的运行警告。")
    lines.extend(
        [
            "",
            "## 8. Claim-Evidence核对",
            "",
            "- Claim：结果来自无数据集级模型调参的统一协议。Evidence：除路径、K和指定batch规则外，全部训练参数一致。Status：supported。",
            "- Claim：某些数据集更适合作为当前模型的后续实验对象。Evidence：本表提供均值与波动，可用于初筛。Status：limited，仍需基线与缺失率实验。",
            "- Claim：当前模型优于已有方法或达到SOTA。Evidence：未运行外部基线。Status：unsupported，本报告不作此声明。",
            "",
            "## 9. 审稿式自检",
            "",
            "- 贡献：提供统一默认协议下的跨数据集适用性证据，但不构成新方法贡献。",
            "- 写作清晰度：路径解析、视图选择、batch规则、种子和统计方式均明确。",
            "- 实验强度：包含5次独立运行，但缺少显著性检验与强基线。",
            "- 评估完整性：缺少全部视图版本、模块消融和多缺失率测试。",
            "- 方法合理性：单视图数据不应通过复制或任意切分伪造成多视图输入。",
            "",
        ]
    )
    (output / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--datasets", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT
    seeds = tuple(args.seeds)
    requested = tuple(args.datasets) if args.datasets else REQUESTED_DATASETS
    summaries: list[dict] = []
    incompatible: list[dict] = []

    for dataname in requested:
        try:
            data = load_dataset(dataname, Config.missing_rate, seeds[0])
        except ValueError as error:
            key = dataname.lower()
            incompatible.append(
                {
                    "dataset": dataname,
                    "source_path": DATASET_PATHS.get(key),
                    "samples": 1440 if key == "coil20" else None,
                    "classes": 20 if key == "coil20" else None,
                    "source_views": 1 if key == "coil20" else None,
                    "batch_size": 256 if key == "coil20" else None,
                    "reason": str(error),
                }
            )
            write_outputs(output, summaries, incompatible, seeds)
            print(f"SKIP dataset={dataname} reason={error}", flush=True)
            continue

        runs = []
        for seed in seeds:
            print(
                f"START dataset={dataname} seed={seed} n={len(data.labels)} batch={data.batch_size} " ,
                flush=True,
            )
            run = run_one(dataname, output, seed)
            runs.append(run)
            print(
                f"DONE dataset={dataname} seed={seed} "
                + " ".join(f"{metric}={run['final'][metric]:.4f}" for metric in METRICS),
                flush=True,
            )
        summary = aggregate(dataname, runs, output)
        summaries.append(summary)
        print(
            f"SUMMARY dataset={summary['dataset']} "
            + " ".join(
                f"{metric}={format_metric(summary['metrics'][metric])}" for metric in METRICS
            ),
            flush=True,
        )
        write_outputs(output, summaries, incompatible, seeds)

    print(f"REPORT {output / 'experiment_report.md'}", flush=True)


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()

    total_time = time.time() - start_time
    print("Running all time: ",total_time)
    print("Average Time: ", total_time / 5)


    # CiteSeer, rate 0.3: 62 64 5 10 25; rate 0.1: 62 15

    # BBCSport rate 0.1 61 63 64 5 15
