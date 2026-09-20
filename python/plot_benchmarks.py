"""Plot medians across measured repetitions; throughput bands show min/max."""
import argparse
import csv
import os
from pathlib import Path
from statistics import median
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlrt-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(csv_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("benchmark CSV has no rows")
    grouped = {}
    for row in rows:
        key = (row["mode"], int(row["threads"]), int(row["batch_size"]))
        grouped.setdefault(key, []).append(row)
    threads = sorted({t for mode, t, _ in grouped if mode == "concurrent"})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    plots = [("throughput_rps", "Throughput (requests/s)", "throughput.png"),
             ("p95_latency_ms", "Caller latency (ms)", "latency.png"),
             ("cpu_seconds", "Process CPU time (seconds)", "cpu_time.png"),
             ("peak_rss_mib", "Process lifetime peak RSS (MiB)", "peak_memory.png")]
    for metric, ylabel, name in plots:
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        for t in threads:
            sizes = sorted(b for mode, n, b in grouped if mode == "concurrent" and n == t)
            values = [[float(r[metric]) for r in grouped[("concurrent", t, b)] if r[metric] != ""] for b in sizes]
            if any(not group for group in values):
                continue
            line, = ax.plot(sizes, [median(group) for group in values], "o-", label=f"{t} worker(s)")
            if metric == "throughput_rps":
                ax.fill_between(sizes, [min(group) for group in values], [max(group) for group in values],
                                color=line.get_color(), alpha=0.12)
            if metric == "p95_latency_ms":
                means = [median(float(r["mean_latency_ms"]) for r in grouped[("concurrent", t, b)]) for b in sizes]
                ax.plot(sizes, means, "--", color=line.get_color(), alpha=0.7)
        baseline = grouped.get(("sync", 1, 1), [])
        baseline_values = [float(r[metric]) for r in baseline if r[metric] != ""]
        if baseline_values:
            ax.axhline(median(baseline_values), color="0.4", linestyle=":", label="sequential caller, 1 worker")
        ax.set(xlabel="Maximum dynamic batch size", ylabel=ylabel, ylim=(0, None))
        ax.set_xticks(sorted({b for mode, _, b in grouped if mode == "concurrent"}))
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        detail = "solid: p95; dashed: mean" if metric == "p95_latency_ms" else "median across repetitions"
        if metric == "throughput_rps":
            detail += "; band: min–max"
        ax.set_title(f"ML Inference Runtime · {ylabel.split(' (')[0]}\n{detail}", loc="left", fontsize=12)
        fig.savefig(output / name)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/local"))
    args = parser.parse_args()
    plot(args.csv, args.output)
