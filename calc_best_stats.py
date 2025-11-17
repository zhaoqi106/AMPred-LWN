import json
import os
import numpy as np
from collections import defaultdict


def collect_metrics(directory: str):
    """Read best_metrics_fold*.txt in directory and return a dict of lists."""
    metrics = defaultdict(list)
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("best_metrics_fold") and fname.endswith(".txt"):
            path = os.path.join(directory, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                # 保存 epoch 也作为浮点数处理，方便求均值
                metrics[key].append(value)
    return metrics


def summarize(metrics: dict):
    """Return dict of metric -> (mean, std)."""
    summary = {}
    for key, values in metrics.items():
        arr = np.array(values, dtype=float)
        summary[key] = (arr.mean(), arr.std(ddof=1))  # 样本标准差
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute mean±std for best_metrics folds")
    parser.add_argument("--dir", default="results/AMES", help="Directory containing best_metrics_fold*.txt")
    parser.add_argument("--precision", type=int, default=3, help="Decimal places for output")
    args = parser.parse_args()

    metrics = collect_metrics(args.dir)
    if not metrics:
        raise FileNotFoundError(f"No best_metrics_fold*.txt found in {args.dir}")

    summary = summarize(metrics)
    fmt = f"{{:.{args.precision}f}}"
    print("Metric\tMean\tStd")
    for key in sorted(summary.keys()):
        mean, std = summary[key]
        print(f"{key}\t{fmt.format(mean)}\t{fmt.format(std)}")


if __name__ == "__main__":
    main()