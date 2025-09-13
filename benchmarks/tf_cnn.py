from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import tensorflow as tf


def _build_model(input_shape: Tuple[int, int, int]) -> "tf.keras.Model":
    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=7)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(inputs, outputs)


def _make_dataset(dataset: str, batch_size: int) -> "Tuple[tf.data.Dataset, Tuple[int, int, int]]":
    import tensorflow as tf

    if dataset == "fashion_mnist":
        (x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
        x = (x[..., None] / 255.0).astype("float32")
        input_shape = (28, 28, 1)
    elif dataset == "cifar10":
        (x, y), _ = tf.keras.datasets.cifar10.load_data()
        y = y.reshape(-1)
        x = (x / 255.0).astype("float32")
        input_shape = (32, 32, 3)
    else:  # pragma: no cover
        raise ValueError(f"Unknown dataset: {dataset}")

    ds = tf.data.Dataset.from_tensor_slices((x, y.astype("int32")))
    ds = ds.shuffle(10_000).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, input_shape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal CPU CNN benchmark with smartclip (TensorFlow)"
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        help="Path to write per-step CSV logs (defaults to runs/tf-<dataset>-<algo>.csv)",
    )
    args = parser.parse_args()

    import tensorflow as tf

    import smartclip as sc
    from smartclip.backends import tf as sc_tf

    tf.config.set_visible_devices([], "GPU")
    ds, input_shape = _make_dataset(args.dataset, args.batch_size)
    it = iter(ds)
    model = _build_model(input_shape)
    opt = tf.keras.optimizers.Adam(args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    clipper = None
    if args.algo == "autoclip":
        clipper = sc.AutoClip()  # auto mode (hyperparameter-free)
    elif args.algo == "agc":
        clipper = sc.AGC(clipping=0.01, exclude_bias_bn=True, scope="per_param")
    elif args.algo == "zscore":
        clipper = sc.ZScoreClip(zmax=3.0, ema_decay=0.99)

    @tf.function(jit_compile=False)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        if clipper is not None:
            grads = sc_tf.apply_grads(grads, model, clipper)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, logits

    import csv
    from pathlib import Path

    start = time.time()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # CSV logging setup
    if not args.log_csv:
        args.log_csv = f"runs/tf-{args.dataset}-{args.algo}.csv"
    out_path = Path(args.log_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = out_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["step", "loss", "acc", "examples", "algo", "dataset", "framework"]
    )  # header

    for step in range(1, args.steps + 1):
        x, y = next(it)
        loss, logits = train_step(x, y)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        total_correct += int(tf.reduce_sum(tf.cast(preds == y, tf.int32)).numpy())
        total_examples += int(tf.shape(y)[0].numpy())
        total_loss += float(loss.numpy())
        csv_writer.writerow(
            [
                step,
                float(loss.numpy()),
                total_correct / max(total_examples, 1),
                total_examples,
                args.algo,
                args.dataset,
                "tf",
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
        if name.startswith("tensorflow"):
            print(
                "TensorFlow is not installed. Install CPU: pip install tensorflow-cpu --only-binary :all:"
            )
        elif name == "smartclip":
            print("Install smartclip: pip install smartclip[tf]")
        else:
            raise
