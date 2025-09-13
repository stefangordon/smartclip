from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Tuple

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def _ensure_dataset(data_dir: Path) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tiny_shakespeare.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    try:
        import urllib.request

        print(f"Downloading Tiny Shakespeare to {path} ...")
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL) as resp:  # nosec - public text file
            text = resp.read().decode("utf-8")
        path.write_text(text, encoding="utf-8")
        return text
    except Exception as exc:  # pragma: no cover
        # Fallback tiny corpus if download not available
        print(f"Warning: download failed ({exc}); using a tiny built-in sample.")
        sample = "To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer...\n"
        path.write_text(sample, encoding="utf-8")
        return sample


def _build_vocab(text: str) -> Tuple[dict[str, int], dict[int, str]]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def _encode(text: str, stoi: dict[str, int]) -> "list[int]":
    return [stoi[ch] for ch in text]


class CharLSTM(torch.nn.Module):  # type: ignore[name-defined]
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden: int = 300, layers: int = 2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(
            input_size=embed_dim, hidden_size=hidden, num_layers=layers, batch_first=True
        )
        self.proj = torch.nn.Linear(hidden, vocab_size)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        e = self.embed(x)
        h, _ = self.lstm(e)
        return self.proj(h)


def _batchify(
    ids: "torch.Tensor",
    batch_size: int,
    seq_len: int,
    gen: "torch.Generator",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    max_start = ids.numel() - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=gen)
    x = torch.stack([ids[s : s + seq_len] for s in starts], dim=0)
    y = torch.stack([ids[s + 1 : s + 1 + seq_len] for s in starts], dim=0)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Char-level LM (Tiny Shakespeare) with smartclip (PyTorch)"
    )
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument(
        "--algo",
        choices=["none", "autoclip", "agc", "zscore"],
        default="autoclip",
        help="Clipping algorithm",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-csv",
        type=str,
        default="",
        help="Path to write per-step CSV logs (defaults to runs/pt-charlm-<algo>-s<seed>.csv)",
    )
    args = parser.parse_args()

    import random

    import numpy as np  # type: ignore
    import torch

    import smartclip as sc

    # Determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
    except Exception:
        pass
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # Data
    raw = _ensure_dataset(Path("data"))
    stoi, _ = _build_vocab(raw)
    vocab_size = len(stoi)
    ids = torch.tensor(_encode(raw, stoi), dtype=torch.long)

    # Model/opt
    model = CharLSTM(vocab_size=vocab_size).to(device)
    # Speed knobs: clip sequence length; use fused AdamW if available
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Clipper
    clipper = None
    if args.algo == "autoclip":
        clipper = sc.AutoClip()  # auto mode
    elif args.algo == "agc":
        clipper = sc.AGC(clipping=0.02, exclude_bias_bn=True, scope="per_layer")
    elif args.algo == "zscore":
        clipper = sc.ZScoreClip(zmax=2.5, ema_decay=0.99)

    # Logging
    if not args.log_csv:
        args.log_csv = f"runs/pt-charlm-{args.algo}-s{args.seed}.csv"
    out_path = Path(args.log_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = out_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "ppl", "algo", "dataset", "framework", "seed"])  # header

    # Train loop
    start = time.time()
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    context = (
        sc.clip_context(model, optimizer=opt, clipper=clipper)
        if clipper is not None
        else nullcontext()
    )

    with context:
        for step in range(1, args.steps + 1):
            x, y = _batchify(ids, args.batch_size, args.seq_len, gen)
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            opt.step()

            ppl = float(math.exp(min(20.0, float(loss.item()))))  # guard overflow
            csv_writer.writerow(
                [step, float(loss.item()), ppl, args.algo, "charlm", "pt", args.seed]
            )

    elapsed = time.time() - start
    print(
        f"algo={args.algo} steps={args.steps} bs={args.batch_size} seq={args.seq_len} "
        f"time_s={elapsed:.2f}"
    )
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
