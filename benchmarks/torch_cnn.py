from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import torch


def _build_model(in_channels: int) -> "torch.nn.Module":
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(in_channels, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10),
    )


def _make_loader(dataset: str, batch_size: int, seed: int) -> "Tuple[object, int]":
    """Returns (loader, in_channels)."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    gen = torch.Generator()
    gen.manual_seed(seed)
    if dataset == "fashion_mnist":
        transform = transforms.ToTensor()
        ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
        in_channels = 1
    elif dataset == "cifar10":
        transform = transforms.ToTensor()
        ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        in_channels = 3
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown dataset: {dataset}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=gen,
    )
    return loader, in_channels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal CPU CNN benchmark with smartclip (PyTorch)"
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads to use (0=auto)")
    parser.add_argument(
        "--dataset",
        choices=["fashion_mnist", "cifar10"],
        default="fashion_mnist",
        help="Dataset to train on",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--algo",
        choices=["none", "autoclip", "agc", "zscore"],
        default="autoclip",
        help="Clipping algorithm",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="",
        help="Path to write per-step CSV logs (defaults to runs/pt-<dataset>-<algo>.csv)",
    )
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim

    import smartclip as sc

    # Deterministic setup for repeatability
    try:
        import random

        import numpy as np  # type: ignore

        random.seed(args.seed)
        np.random.seed(args.seed)
    except Exception:
        pass
    torch.manual_seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
    except Exception:
        pass
    threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    try:
        torch.set_num_threads(int(threads))
        torch.set_num_interop_threads(max(1, min(int(threads), 4)))
    except Exception:
        pass
    device = torch.device("cpu")

    loader, in_channels = _make_loader(args.dataset, args.batch_size, args.seed)
    model = _build_model(in_channels).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    clipper = None
    if args.algo == "autoclip":
        # Percentile mode for clearer demo separation
        clipper = sc.AutoClip(
            mode="percentile",
            percentile=90.0,
            history="ema",
            ema_decay=0.99,
            warmup_steps=0,
            min_history=10,
        )
    elif args.algo == "agc":
        # Stronger clipping to accentuate differences
        clipping_val = 0.05 if args.dataset == "cifar10" else 0.03
        clipper = sc.AGC(clipping=clipping_val, exclude_bias_bn=True, scope="per_layer")
    elif args.algo == "zscore":
        # Tighter bounds to trigger more often
        zmax_val = 2.0
        clipper = sc.ZScoreClip(zmax=zmax_val, ema_decay=0.99)

    start = time.time()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = (
        sc.clip_context(model, optimizer=opt, clipper=clipper)
        if clipper is not None
        else nullcontext()
    )

    csv_writer = None
    csv_file = None
    if not args.log_csv:
        args.log_csv = f"runs/pt-{args.dataset}-{args.algo}-s{args.seed}.csv"
    out_path = Path(args.log_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = out_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["step", "loss", "acc", "examples", "algo", "dataset", "framework", "seed"]
    )  # header

    it = iter(loader)
    with context:
        for step in range(1, args.steps + 1):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                total_correct += int((preds == y).sum().item())
                total_examples += y.numel()
            batch_loss = float(loss.item())
            total_loss += batch_loss
            if csv_writer is not None:
                acc_step = total_correct / max(total_examples, 1)
                csv_writer.writerow(
                    [
                        step,
                        batch_loss,
                        acc_step,
                        total_examples,
                        args.algo,
                        args.dataset,
                        "pt",
                        args.seed,
                    ]
                )

    elapsed = time.time() - start
    throughput = total_examples / max(elapsed, 1e-9)
    avg_loss = total_loss / float(args.steps)
    acc = total_correct / max(total_examples, 1)

    algo_name = args.algo
    print(
        f"algo={algo_name} steps={args.steps} bs={args.batch_size} "
        f"avg_loss={avg_loss:.4f} acc={acc:.3f} ex/s={throughput:.1f} time_s={elapsed:.2f}"
    )

    if csv_file is not None:
        csv_file.close()


class nullcontext:
    def __enter__(self):  # type: ignore[no-untyped-def]
        return None

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        return False


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:  # pragma: no cover
        name = getattr(exc, "name", "")
        if name == "torch":
            print(
                "PyTorch is not installed. Install CPU wheels: pip install --index-url https://download.pytorch.org/whl/cpu torch"
            )
        elif name == "smartclip":
            print("Install smartclip: pip install smartclip[torch]")
        else:
            raise
