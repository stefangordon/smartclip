from __future__ import annotations

import argparse
import time
from typing import Any, Tuple


def _init_params(rng: Any, in_dim: int, hidden: int, out_dim: int):
    import jax
    import jax.numpy as jnp

    k1, k2, k3, k4 = jax.random.split(rng, 4)
    params = {
        "w1": jnp.asarray(0.01) * jax.random.normal(k1, (in_dim, hidden)),
        "b1": jnp.zeros((hidden,)),
        "w2": jnp.asarray(0.01) * jax.random.normal(k2, (hidden, out_dim)),
        "b2": jnp.zeros((out_dim,)),
    }
    return params


def _forward(params: dict[str, Any], x: Any) -> Any:
    import jax.numpy as jnp

    h = jnp.tanh(x @ params["w1"] + params["b1"])  # type: ignore[no-any-return]
    logits = h @ params["w2"] + params["b2"]  # type: ignore[no-any-return]
    return logits


def _make_iterator(dataset: str, batch_size: int, in_dim: int) -> Tuple[object, int]:
    """Return (iterator, out_dim)."""
    import jax.numpy as jnp
    import numpy as np

    if dataset == "fashion_mnist":
        try:
            import tensorflow_datasets as tfds  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ModuleNotFoundError("tensorflow_datasets is required for JAX benchmarks") from exc
        ds = tfds.load("fashion_mnist", split="train", as_supervised=True)
        ds = ds.shuffle(10_000).repeat().batch(batch_size)

        def gen():
            for bx, by in tfds.as_numpy(ds):
                x = (bx.astype(np.float32) / 255.0).reshape(bx.shape[0], -1)
                y = by.astype(np.int32)
                yield jnp.asarray(x), jnp.asarray(y)

        return gen(), 10
    elif dataset == "cifar10":
        try:
            import tensorflow_datasets as tfds  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ModuleNotFoundError("tensorflow_datasets is required for JAX benchmarks") from exc
        ds = tfds.load("cifar10", split="train", as_supervised=True)
        ds = ds.shuffle(10_000).repeat().batch(batch_size)

        def gen():
            for bx, by in tfds.as_numpy(ds):
                x = (bx.astype(np.float32) / 255.0).reshape(bx.shape[0], -1)
                y = by.astype(np.int32)
                yield jnp.asarray(x), jnp.asarray(y)

        return gen(), 10
    else:  # pragma: no cover
        raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal CPU MLP benchmark with smartclip (JAX)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--dataset",
        choices=["fashion_mnist", "cifar10"],
        default="fashion_mnist",
        help="Dataset to train on",
    )
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
        help="Path to write per-step CSV logs (defaults to runs/jax-<dataset>-<algo>.csv)",
    )
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import optax

    import smartclip as sc
    from smartclip.backends import jax as sc_jax

    in_dim = 784 if args.dataset == "fashion_mnist" else 32 * 32 * 3
    hidden = 128
    iterator, out_dim = _make_iterator(args.dataset, args.batch_size, in_dim)
    key = jax.random.PRNGKey(0)
    params = _init_params((key, key, key, key), in_dim, hidden, out_dim)
    tx = optax.sgd(args.lr)
    opt_state = tx.init(params)

    def loss_fn(p, x, y):  # type: ignore[no-untyped-def]
        logits = _forward(p, x)
        onehot = jax.nn.one_hot(y, out_dim)
        log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        return -jnp.mean(jnp.sum(onehot * log_probs, axis=-1))

    grad_fn = jax.jit(jax.grad(loss_fn))

    clipper = None
    if args.algo == "autoclip":
        clipper = sc.AutoClip()  # auto mode (hyperparameter-free)
    elif args.algo == "agc":
        clipper = sc.AGC(clipping=0.01, exclude_bias_bn=True, scope="per_param")
    elif args.algo == "zscore":
        clipper = sc.ZScoreClip(zmax=3.0, ema_decay=0.99)

    import csv
    from pathlib import Path

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    start = time.time()

    # CSV logging setup
    if not args.log_csv:
        args.log_csv = f"runs/jax-{args.dataset}-{args.algo}.csv"
    out_path = Path(args.log_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = out_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["step", "loss", "acc", "examples", "algo", "dataset", "framework"]
    )  # header

    for step in range(1, args.steps + 1):
        x, y = next(iterator)
        grads = grad_fn(params, x, y)
        if clipper is not None:
            grads = sc_jax.apply_grads(grads, params, clipper)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        logits = _forward(params, x)
        preds = jnp.argmax(logits, axis=-1)
        total_correct += int(jnp.sum(preds == y))
        total_examples += int(y.shape[0])

        # Compute loss from logits to avoid redundant forward pass
        onehot = jax.nn.one_hot(y, out_dim)
        log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        batch_loss = -jnp.mean(jnp.sum(onehot * log_probs, axis=-1))
        total_loss += float(batch_loss)
        csv_writer.writerow(
            [
                step,
                float(batch_loss),
                total_correct / max(total_examples, 1),
                total_examples,
                args.algo,
                args.dataset,
                "jax",
            ]
        )

    elapsed = time.time() - start
    throughput = total_examples / max(elapsed, 1e-9)
    avg_loss = total_loss / float(args.steps)
    acc = total_correct / max(total_examples, 1)
    print(
        f"algo={args.algo} steps={args.steps} bs={args.batch_size} "
        f"avg_loss={avg_loss:.4f} acc={acc:.3f} ex/s={throughput:.1f} time_s={elapsed:.2f}"
    )

    csv_file.close()


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:  # pragma: no cover
        name = getattr(exc, "name", "")
        if name == "jax":
            print("JAX is not installed. Install CPU: pip install 'jax[cpu]' optax")
        elif name == "smartclip":
            print("Install smartclip: pip install smartclip[jax]")
        else:
            raise
