from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_series(csv_path: Path) -> Dict[str, List[float]]:
    steps: List[int] = []
    losses: List[float] = []
    accs: List[float] = []
    algo_name = "unknown"
    dataset = "unknown"
    framework = "unknown"
    metric_label = "accuracy"
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            if "acc" in row and row["acc"] != "":
                accs.append(float(row["acc"]))
                metric_label = "accuracy"
            elif "ppl" in row and row["ppl"] != "":
                accs.append(float(row["ppl"]))
                metric_label = "perplexity"
            algo_name = row.get("algo", algo_name)
            dataset = row.get("dataset", dataset)
            framework = row.get("framework", framework)
    return {
        "algo": algo_name,
        "dataset": dataset,
        "framework": framework,
        "step": steps,
        "loss": losses,
        "acc": accs,
        "metric_label": metric_label,
    }


def _run_pt_benchmarks(
    datasets: List[str],
    algos: List[str],
    steps: int,
    batch_size: int,
    seeds: List[int],
    threads: int | None = None,
) -> None:
    import os
    import subprocess
    import sys

    for ds in datasets:
        for algo in algos:
            # Higher LR to accentuate clipping benefits (FMNIST higher too)
            lr = "2e-2" if ds == "cifar10" else "5e-3"
            for seed in seeds:
                cmd = [
                    sys.executable,
                    "benchmarks/torch_cnn.py",
                    "--steps",
                    str(steps),
                    "--batch-size",
                    str(batch_size),
                    "--dataset",
                    ds,
                    "--algo",
                    algo,
                    "--lr",
                    lr,
                    "--seed",
                    str(seed),
                ]
                print("Running:", " ".join(cmd))
                env = dict(**os.environ)
                if threads and threads > 0:
                    t = str(threads)
                    env.update(
                        {
                            "OMP_NUM_THREADS": t,
                            "MKL_NUM_THREADS": t,
                            "OPENBLAS_NUM_THREADS": t,
                            "NUMEXPR_NUM_THREADS": t,
                        }
                    )
                    cmd += ["--threads", t]
                subprocess.run(cmd, check=True, env=env)


def _run_pt_charlm(
    algos: List[str], steps: int, batch_size: int, seeds: List[int], seq_len: int = 128
) -> None:
    import subprocess
    import sys

    for algo in algos:
        for seed in seeds:
            cmd = [
                sys.executable,
                "benchmarks/torch_charlm.py",
                "--steps",
                str(steps),
                "--batch-size",
                str(batch_size),
                "--seq-len",
                str(seq_len),
                "--algo",
                algo,
                "--seed",
                str(seed),
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmarks and plot SVGs (PyTorch CNN, FMNIST & CIFAR-10)"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-run", action="store_true", help="Skip running benchmarks; only plot existing CSVs"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=[0, 1, 2], help="Seeds to average over"
    )
    parser.add_argument(
        "--output", type=str, default="docs/assets/benchmarks-{framework}-{dataset}.svg"
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="CPU threads to use for benchmarks (0=auto)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print debug info and write averaged CSVs"
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    algos = ["none", "autoclip", "agc", "zscore"]

    if not args.skip_run:
        # Clean up old CSV files before running new benchmarks
        runs_dir = Path("runs")
        if runs_dir.exists():
            for csv_file in runs_dir.glob("*.csv"):
                csv_file.unlink()
                print(f"Removed old CSV: {csv_file}")
        # Run vision (FMNIST/CIFAR-10)
        threads = args.threads if args.threads > 0 else None
        _run_pt_benchmarks(
            ["fashion_mnist", "cifar10"], algos, args.steps, args.batch_size, args.seeds, threads
        )
        # Run character LM (Tiny Shakespeare)
        _run_pt_charlm(algos, args.steps, args.batch_size, args.seeds, seq_len=128)

    inputs = sorted(Path("runs").glob("*.csv"))
    series_raw = [read_series(p) for p in inputs]

    # Group by (framework, dataset, algo) and average across seeds
    from statistics import mean

    grouped: Dict[tuple[str, str, str], Dict[str, List[float]]] = {}
    for s in series_raw:
        key = (
            str(s.get("framework", "unknown")),
            str(s.get("dataset", "unknown")),
            str(s.get("algo", "unknown")),
        )
        grouped.setdefault(key, {"step": [], "loss": [], "acc": [], "metric_label": []})
        # Assume equal length per run; append series for later averaging
        grouped[key]["step"].append(s["step"])  # type: ignore[arg-type]
        grouped[key]["loss"].append(s["loss"])  # type: ignore[arg-type]
        grouped[key]["acc"].append(s["acc"])  # type: ignore[arg-type]
        grouped[key]["metric_label"].append(str(s.get("metric_label", "accuracy")))

    # Build averaged series by (framework, dataset)
    by_group: Dict[tuple[str, str], List[Dict[str, List[float]]]] = {}
    for (framework, dataset, algo), vals in grouped.items():
        # Average pointwise across seeds
        steps_list = vals["step"]
        loss_list = vals["loss"]
        acc_list = vals["acc"]
        steps = steps_list[0]
        avg_loss = [mean(xs) for xs in zip(*loss_list)]
        avg_acc = [mean(xs) for xs in zip(*acc_list)]
        label2 = vals["metric_label"][0] if vals["metric_label"] else "accuracy"
        by_group.setdefault((framework, dataset), []).append(
            {
                "algo": algo,
                "step": steps,
                "loss": avg_loss,
                "acc": avg_acc,
                "metric_label": label2,
            }
        )

        if args.debug:
            # Emit a small summary
            print(
                f"AVG {framework}/{dataset}/{algo}: steps={len(steps)} loss[0]={avg_loss[0]:.4f} loss[-1]={avg_loss[-1]:.4f}"
            )
            # Write averaged CSV for manual inspection
            out_avg = Path("runs") / f"avg-{framework}-{dataset}-{algo}.csv"
            out_avg.parent.mkdir(parents=True, exist_ok=True)
            with out_avg.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["step", "loss", "metric", "algo", "dataset", "framework"]
                )  # header
                for s, loss_val, metric_val in zip(steps, avg_loss, avg_acc):
                    writer.writerow([s, loss_val, label2, algo, dataset, framework])

    wrote_any = False
    for (framework, dataset), series in by_group.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)

        # Loss vs step
        for s in series:
            axes[0].plot(s["step"], s["loss"], label=s["algo"])  # type: ignore[arg-type]
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].set_title(f"Loss ({framework}, {dataset})")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Second panel: accuracy or perplexity vs step
        for s in series:
            axes[1].plot(s["step"], s["acc"], label=s["algo"])  # type: ignore[arg-type]
        axes[1].set_xlabel("step")
        y2 = series[0].get("metric_label", "accuracy")
        axes[1].set_ylabel(y2)
        axes[1].set_title(f"{y2.title()} ({framework}, {dataset})")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        out_path = Path(args.output.format(framework=framework, dataset=dataset))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, format="svg")
        print(f"Wrote plot to {out_path}")
        wrote_any = True

    if not wrote_any:
        print("No series found. Ensure CSVs exist under runs/*.csv or pass --inputs.")


if __name__ == "__main__":
    main()
