"""C1 — Neural networks from first principles: perceptron, hand-written backprop, and ablations.

    python -m lab_c.fundamentals --config lab-c-vision/configs/c1_fundamentals.yaml

1. The perceptron learns any linearly separable problem and never learns XOR;
   a two-layer network does.
2. A NumPy MLP with hand-written backpropagation passes a finite-difference
   gradient check and matches PyTorch's autograd to machine precision; trained
   side by side with identical weights and batches, both learn the same function.
3. Ablations of a deep MLP on Fashion-MNIST: activation, initialisation,
   optimiser, batch normalisation and dropout, over several seeds.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from airlab.config import ExperimentConfig
from airlab.experiment import Outputs, cli
from airlab.metrics import bootstrap_ci
from airlab.plots import plt, save
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import mcnemar
from airlab.tracking import track_run
from lab_c.datasets import ImageSet, load_images
from lab_c.nets import MLP, Activation, Init
from lab_c.scratch import NumpyMLP, Perceptron, gradient_check
from lab_c.train import TrainConfig, fit, predict

DEFAULT_OUTPUT_DIR = Path("lab-c-vision/reports")


class Ablation(BaseModel):
    name: str
    activation: Activation = "relu"
    init: Init = "default"
    optimizer: Literal["sgd", "momentum", "adam", "adamw"] = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0  # AdamW only differs from Adam through decoupled weight decay
    batchnorm: bool = False
    dropout: float = 0.0


def _default_ablations() -> dict[str, list[Ablation]]:
    return {
        "Activation": [Ablation(name=a, activation=a) for a in ("relu", "gelu", "tanh", "sigmoid")],
        "Initialisation (ReLU)": [
            Ablation(name="PyTorch default", init="default"),
            Ablation(name="N(0, 0.01²)", init="small_normal"),
            Ablation(name="Xavier", init="xavier"),
            Ablation(name="He", init="he"),
        ],
        "Optimiser": [
            Ablation(name="SGD", optimizer="sgd", lr=0.05),
            Ablation(name="SGD + Nesterov momentum", optimizer="momentum", lr=0.05),
            Ablation(name="Adam", optimizer="adam", lr=1e-3),
            Ablation(
                name="AdamW (weight decay 0.05)", optimizer="adamw", lr=1e-3, weight_decay=0.05
            ),
        ],
        "Normalisation & regularisation": [
            Ablation(name="none"),
            Ablation(name="batch norm", batchnorm=True),
            Ablation(name="dropout 0.2", dropout=0.2),
            Ablation(name="dropout 0.5", dropout=0.5),
            Ablation(name="batch norm + dropout 0.2", batchnorm=True, dropout=0.2),
        ],
    }


class FundamentalsConfig(ExperimentConfig):
    output_dir: Path = DEFAULT_OUTPUT_DIR
    dataset: str = "fashion_mnist"
    max_train: int | None = None
    max_test: int | None = None
    numpy_hidden: list[int] = Field(default_factory=lambda: [128, 64])
    numpy_epochs: int = 5
    numpy_lr: float = 0.05
    deep_hidden: list[int] = Field(default_factory=lambda: [256, 256, 256, 256])
    ablation_epochs: int = 5
    seeds: int = 3
    ablations: dict[str, list[Ablation]] = Field(default_factory=_default_ablations)
    bootstrap_resamples: int = 1000


# --------------------------------------------------------------------- perceptron


def perceptron_section(cfg: FundamentalsConfig, out: Outputs) -> list[str]:
    rng = np.random.default_rng(cfg.seed)
    X_sep = rng.normal(0, 1, (200, 2))
    y_sep = (X_sep[:, 0] + 2 * X_sep[:, 1] > 0.3).astype(int)
    X_xor = np.repeat(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float), 50, axis=0) + rng.normal(
        0, 0.12, (200, 2)
    )
    y_xor = np.repeat(np.array([0, 1, 1, 0]), 50)

    p_sep = Perceptron(max_epochs=100, seed=cfg.seed).fit(X_sep, y_sep)
    p_xor = Perceptron(max_epochs=100, seed=cfg.seed).fit(X_xor, y_xor)
    mlp = NumpyMLP([2, 8, 2], "tanh", seed=cfg.seed)
    for epoch in range(300):
        mlp.fit_epoch(X_xor, y_xor, lr=0.1, batch_size=16, seed=epoch)
    acc = {
        "perceptron, separable": float((p_sep.predict(X_sep) == y_sep).mean()),
        "perceptron, XOR": float((p_xor.predict(X_xor) == y_xor).mean()),
        "2-layer MLP (8 tanh units), XOR": float((mlp.predict(X_xor) == y_xor).mean()),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [
        (X_sep, y_sep, p_sep.predict, f"Perceptron, separable: {len(p_sep.errors)} epochs"),
        (X_xor, y_xor, p_xor.predict, "Perceptron, XOR: never converges"),
        (X_xor, y_xor, mlp.predict, "NumPy MLP, XOR"),
    ]
    for ax, (X, y, predict_fn, title) in zip(axes, panels, strict=True):
        lo, hi = X.min(0) - 0.5, X.max(0) + 0.5
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
        zz = predict_fn(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx, gy, zz, alpha=0.25, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10, edgecolors="none")
        ax.set(title=title, xticks=[], yticks=[])
    path, link = out.figure("decision_boundaries.png")
    save(fig, path)
    return [
        "## Perceptron vs. a hidden layer",
        "",
        md_table(["Model / data", "Training accuracy"], [[k, f"{v:.1%}"] for k, v in acc.items()]),
        "",
        f"The perceptron stops after {len(p_sep.errors)} epochs on separable data and is still making "
        f"{p_xor.errors[-1]} mistakes per epoch on XOR after {len(p_xor.errors)}: no line separates XOR, "
        "while one hidden layer bends the boundary.",
        "",
        f"![decision boundaries]({link})",
        "",
    ]


# ---------------------------------------------------------------- NumPy vs. torch


def _flat(data: ImageSet) -> tuple[np.ndarray, np.ndarray]:
    return (
        data.images.reshape(len(data), -1).numpy().astype(np.float64) / 255 - 0.5
    ) / 0.5, data.labels.numpy()


def numpy_section(
    train: ImageSet, test: ImageSet, cfg: FundamentalsConfig, out: Outputs
) -> tuple[list[str], list]:
    X_tr, y_tr = _flat(train)
    X_te, y_te = _flat(test)
    sizes = [X_tr.shape[1], *cfg.numpy_hidden, train.n_classes]

    # gradient check and autograd agreement on a small batch
    probe = NumpyMLP(sizes, "relu", seed=cfg.seed)
    rel = gradient_check(probe, X_tr[:64], y_tr[:64], n_checks=15, seed=cfg.seed)
    loss_np, grads_np = probe.loss_and_grads(X_tr[:64], y_tr[:64])
    net = probe.to_torch()
    loss_t = torch.nn.functional.cross_entropy(
        net(torch.from_numpy(X_tr[:64])), torch.from_numpy(y_tr[:64])
    )
    loss_t.backward()
    grads_t = [p.grad.numpy() for p in net.parameters()]
    autograd_diff = max(
        float(np.abs(g - (gt.T if g.ndim == 2 else gt)).max())
        for g, gt in zip(grads_np, grads_t, strict=True)
    )

    # train both from the same weights on the same batches
    numpy_model = NumpyMLP(sizes, "relu", seed=cfg.seed)
    torch_model = numpy_model.to_torch()
    opt = torch.optim.SGD(torch_model.parameters(), lr=cfg.numpy_lr, momentum=0.9)
    curves: dict[str, list[float]] = {"NumPy": [], "PyTorch": []}
    seconds: dict[str, float] = {"NumPy": 0.0, "PyTorch": 0.0}
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    for epoch in range(cfg.numpy_epochs):
        start = time.perf_counter()
        curves["NumPy"].append(numpy_model.fit_epoch(X_tr, y_tr, lr=cfg.numpy_lr, seed=epoch))
        seconds["NumPy"] += time.perf_counter() - start
        start = time.perf_counter()
        order = np.random.default_rng(epoch).permutation(len(X_tr))
        losses = []
        for i in range(0, len(order), 128):
            idx = torch.from_numpy(order[i : i + 128])
            loss = torch.nn.functional.cross_entropy(torch_model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        curves["PyTorch"].append(float(np.mean(losses)))
        seconds["PyTorch"] += time.perf_counter() - start

    with torch.no_grad():
        pred_t = torch_model(torch.from_numpy(X_te)).argmax(1).numpy()
    pred_np = numpy_model.predict(X_te)
    acc = {
        k: bootstrap_ci(
            y_te,
            p,
            lambda a, b: float(np.mean(a == b)),
            n_resamples=cfg.bootstrap_resamples,
            seed=cfg.seed,
        )
        for k, p in (("NumPy", pred_np), ("PyTorch", pred_t))
    }
    agree = float(np.mean(pred_np == pred_t))

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for name, style in (("NumPy", "-o"), ("PyTorch", "--x")):
        ax.plot(range(1, cfg.numpy_epochs + 1), curves[name], style, label=name)
    ax.set(xlabel="epoch", ylabel="training loss", title="Same weights, same batches")
    ax.legend()
    path, link = out.figure("numpy_vs_torch.png")
    save(fig, path)

    layer_names = [f"{'W' if i % 2 == 0 else 'b'}{i // 2 + 1}" for i in range(len(rel))]
    rows = [
        {
            "experiment": cfg.experiment,
            "dataset": train.name,
            "model": k,
            "fold": "test",
            "metric": "accuracy",
            "value": v.value,
        }
        for k, v in acc.items()
    ]
    lines = [
        "## Backpropagation by hand",
        "",
        f"A {' → '.join(map(str, sizes))} ReLU network written in NumPy (forward pass, softmax "
        "cross-entropy, backpropagation and SGD with momentum).",
        "",
        md_table(
            ["Check", "Result"],
            [
                [
                    "Gradient check, worst relative error per tensor",
                    ", ".join(f"{n} {e:.1e}" for n, e in zip(layer_names, rel, strict=True)),
                ],
                [
                    "Loss, NumPy vs. PyTorch on the same batch",
                    f"{loss_np:.10f} vs. {loss_t.item():.10f}",
                ],
                ["Largest gradient difference vs. PyTorch autograd", f"{autograd_diff:.1e}"],
            ],
        ),
        "",
        f"Trained side by side for {cfg.numpy_epochs} epochs from identical weights on identical batches:",
        "",
        md_table(
            ["Implementation", "Test accuracy (95% CI)", "Seconds / epoch"],
            [
                [k, str(acc[k]), f"{seconds[k] / cfg.numpy_epochs:.1f}"]
                for k in ("NumPy", "PyTorch")
            ],
        ),
        "",
        f"The two networks predict the same class for {agree:.2%} of test images.",
        "",
        f"![numpy vs torch]({link})",
        "",
    ]
    return lines, rows


# ----------------------------------------------------------------------- ablations


def ablation_section(
    train: ImageSet, test: ImageSet, cfg: FundamentalsConfig, out: Outputs
) -> tuple[list[str], list]:
    in_dim = int(np.prod(train.images.shape[1:]))
    rows: list[dict[str, Any]] = []
    lines = [
        "## Ablations of a deep MLP",
        "",
        f"{' → '.join(map(str, [in_dim, *cfg.deep_hidden, train.n_classes]))} network, "
        f"{cfg.ablation_epochs} epochs, {cfg.seeds} seeds; one factor changed at a time from "
        "ReLU + PyTorch default init + Adam. McNemar: seed-0 predictions vs. the group's best.",
        "",
    ]
    fig, axes = plt.subplots(
        1, len(cfg.ablations), figsize=(4.2 * len(cfg.ablations), 3.4), sharey=True
    )
    for ax, (group, variants) in zip(np.atleast_1d(axes), cfg.ablations.items(), strict=True):
        results: dict[str, dict[str, Any]] = {}
        for v in variants:
            accs, preds, curves = [], None, []
            for seed in range(cfg.seeds):
                torch.manual_seed(seed)
                model = MLP(
                    in_dim,
                    cfg.deep_hidden,
                    train.n_classes,
                    activation=v.activation,
                    batchnorm=v.batchnorm,
                    dropout=v.dropout,
                    init=v.init,
                )
                tc = TrainConfig(
                    epochs=cfg.ablation_epochs,
                    lr=v.lr,
                    optimizer=v.optimizer,
                    weight_decay=v.weight_decay,
                    seed=seed,
                )
                history = fit(model, train, tc)
                pred = predict(model, test).argmax(1).numpy()
                accs.append(float(np.mean(pred == test.labels.numpy())))
                curves.append(history.train_loss)
                if seed == 0:
                    preds = pred
                rows.append(
                    {
                        "experiment": cfg.experiment,
                        "dataset": f"{train.name} / {group}",
                        "model": v.name,
                        "fold": seed,
                        "metric": "accuracy",
                        "value": accs[-1],
                    }
                )
            results[v.name] = {
                "mean": np.mean(accs),
                "std": np.std(accs, ddof=1) if len(accs) > 1 else 0.0,
                "pred": preds,
            }
            ax.plot(
                range(1, cfg.ablation_epochs + 1),
                np.mean(curves, axis=0),
                marker="o",
                ms=3,
                label=v.name,
            )
        ax.set(title=group, xlabel="epoch", yscale="log")
        ax.legend(fontsize=7)
        best = max(results, key=lambda k: results[k]["mean"])
        y_te = test.labels.numpy()
        lines += [
            f"**{group}**",
            "",
            md_table(
                ["Variant", f"Test accuracy (mean ± std, {cfg.seeds} seeds)", "McNemar p vs best"],
                [
                    [
                        f"**{k}**" if k == best else k,
                        f"{r['mean']:.4f} ± {r['std']:.4f}",
                        "—"
                        if k == best
                        else fmt_p(mcnemar(y_te, results[best]["pred"], r["pred"]).pvalue),
                    ]
                    for k, r in sorted(results.items(), key=lambda kv: -kv[1]["mean"])
                ],
            ),
            "",
        ]
    np.atleast_1d(axes)[0].set_ylabel("training loss (log)")
    path, link = out.figure("ablation_curves.png")
    save(fig, path)
    lines += [f"![training curves]({link})", ""]
    return lines, rows


def run(cfg: FundamentalsConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    train = load_images(cfg.dataset, "train")
    test = load_images(cfg.dataset, "test")
    if cfg.max_train:
        train = train.stratified_fraction(cfg.max_train / len(train), cfg.seed)
    if cfg.max_test:
        test = test.stratified_fraction(cfg.max_test / len(test), cfg.seed)
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        sections = perceptron_section(cfg, out)
        lines, rows_np = numpy_section(train, test, cfg, out)
        sections += lines
        lines, rows_ab = ablation_section(train, test, cfg, out)
        sections += lines
        write_results(rows_np + rows_ab, out.results)
        tracker.log_metrics({f"{r['model']}.accuracy": r["value"] for r in rows_np})
        header = [
            "# C1 — Neural networks from first principles",
            "",
            f"{train.source} · {len(train):,} training / {len(test):,} test images · "
            f"results in [`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv)",
            "",
        ]
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, FundamentalsConfig, "C1 neural-network fundamentals")


if __name__ == "__main__":
    main()
