"""Neural networks from first principles, in NumPy only.

* ``Perceptron`` — Rosenblatt's learning rule; converges iff the data are linearly separable.
* ``NumpyMLP``   — fully connected network with ReLU/tanh hidden layers, softmax
  cross-entropy, hand-written backpropagation and SGD with momentum.
* ``gradient_check`` — compares the analytic gradients with central finite differences.

The PyTorch equivalent (``to_torch``) receives the same weights, so the two
implementations can be checked against each other gradient by gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Activation = Literal["relu", "tanh"]


# ---------------------------------------------------------------------- perceptron


@dataclass
class Perceptron:
    lr: float = 1.0
    max_epochs: int = 100
    seed: int = 0
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b: float = 0.0
    errors: list[int] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Perceptron:
        """``y`` in {0, 1}. Stops at the first epoch without mistakes."""
        rng = np.random.default_rng(self.seed)
        self.w, self.b, self.errors = np.zeros(X.shape[1]), 0.0, []
        for _ in range(self.max_epochs):
            mistakes = 0
            for i in rng.permutation(len(X)):
                update = self.lr * (y[i] - self.predict(X[i : i + 1])[0])
                if update:
                    self.w += update * X[i]
                    self.b += update
                    mistakes += 1
            self.errors.append(mistakes)
            if mistakes == 0:
                break
        return self

    @property
    def converged(self) -> bool:
        return bool(self.errors) and self.errors[-1] == 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.w + self.b > 0).astype(int)


# -------------------------------------------------------------------------- MLP


def _act(z: np.ndarray, kind: Activation) -> np.ndarray:
    return np.maximum(z, 0) if kind == "relu" else np.tanh(z)


def _act_grad(z: np.ndarray, a: np.ndarray, kind: Activation) -> np.ndarray:
    return (z > 0).astype(z.dtype) if kind == "relu" else 1 - a**2


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class NumpyMLP:
    sizes: list[int]
    activation: Activation = "relu"
    seed: int = 0
    dtype: type = np.float64
    params: list[np.ndarray] = field(default_factory=list)  # [W1, b1, W2, b2, ...]
    _velocity: list[np.ndarray] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.params:
            rng = np.random.default_rng(self.seed)
            for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:], strict=True):
                # He initialisation for ReLU, Xavier/Glorot for tanh
                std = np.sqrt(2 / fan_in) if self.activation == "relu" else np.sqrt(1 / fan_in)
                self.params += [
                    rng.normal(0, std, (fan_in, fan_out)).astype(self.dtype),
                    np.zeros(fan_out, self.dtype),
                ]
        self._velocity = [np.zeros_like(p) for p in self.params]

    @property
    def n_layers(self) -> int:
        return len(self.params) // 2

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Logits and, per layer, (pre-activation, activation) for backprop."""
        cache = []
        a = X
        for layer in range(self.n_layers):
            W, b = self.params[2 * layer], self.params[2 * layer + 1]
            z = a @ W + b
            a = z if layer == self.n_layers - 1 else _act(z, self.activation)
            cache.append((z, a))
        return a, cache

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """Mean softmax cross-entropy and its gradient w.r.t. every parameter."""
        logits, cache = self.forward(X)
        probs = softmax(logits)
        n = len(y)
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(
            np.exp(shifted).sum(axis=1, keepdims=True)
        )  # stable log-softmax
        loss = float(-log_probs[np.arange(n), y].mean())

        grads: list[np.ndarray] = [np.empty(0)] * len(self.params)
        delta = probs.copy()
        delta[np.arange(n), y] -= 1  # dL/dlogits for softmax + cross-entropy
        delta /= n
        for layer in reversed(range(self.n_layers)):
            a_prev = X if layer == 0 else cache[layer - 1][1]
            grads[2 * layer] = a_prev.T @ delta
            grads[2 * layer + 1] = delta.sum(axis=0)
            if layer:
                z_prev, a_prev_act = cache[layer - 1]
                delta = (delta @ self.params[2 * layer].T) * _act_grad(
                    z_prev, a_prev_act, self.activation
                )
        return loss, grads

    def step(self, grads: list[np.ndarray], lr: float, momentum: float = 0.9) -> None:
        for i, (p, g) in enumerate(zip(self.params, grads, strict=True)):
            self._velocity[i] = momentum * self._velocity[i] - lr * g
            p += self._velocity[i]

    def fit_epoch(
        self, X: np.ndarray, y: np.ndarray, *, lr: float, batch_size: int = 128, seed: int = 0
    ) -> float:
        order = np.random.default_rng(seed).permutation(len(X))
        losses = []
        for start in range(0, len(X), batch_size):
            idx = order[start : start + batch_size]
            loss, grads = self.loss_and_grads(X[idx], y[idx])
            self.step(grads, lr)
            losses.append(loss)
        return float(np.mean(losses))

    def predict(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        return np.concatenate(
            [
                self.forward(X[i : i + batch_size])[0].argmax(axis=1)
                for i in range(0, len(X), batch_size)
            ]
        )

    def to_torch(self):
        """An identical PyTorch network carrying the same weights."""
        import torch
        from torch import nn

        layers: list[nn.Module] = []
        for layer in range(self.n_layers):
            W, b = self.params[2 * layer], self.params[2 * layer + 1]
            linear = nn.Linear(W.shape[0], W.shape[1]).double()
            with torch.no_grad():
                linear.weight.copy_(torch.from_numpy(W.T.astype(np.float64)))
                linear.bias.copy_(torch.from_numpy(b.astype(np.float64)))
            layers.append(linear)
            if layer < self.n_layers - 1:
                layers.append(nn.ReLU() if self.activation == "relu" else nn.Tanh())
        return nn.Sequential(*layers)


def gradient_check(
    model: NumpyMLP,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_checks: int = 20,
    eps: float = 1e-6,
    seed: int = 0,
) -> list[float]:
    """Relative error between analytic and central-difference gradients, per parameter tensor.

    Values around 1e-7 or below mean the backpropagation is correct (float64).
    """
    rng = np.random.default_rng(seed)
    _, grads = model.loss_and_grads(X, y)
    errors = []
    for p, g in zip(model.params, grads, strict=True):
        worst = 0.0
        for _ in range(n_checks):
            i = tuple(rng.integers(0, s) for s in p.shape)
            original = p[i]
            p[i] = original + eps
            plus, _ = model.loss_and_grads(X, y)
            p[i] = original - eps
            minus, _ = model.loss_and_grads(X, y)
            p[i] = original
            numeric = (plus - minus) / (2 * eps)
            denom = max(abs(numeric) + abs(g[i]), 1e-12)
            worst = max(worst, abs(numeric - g[i]) / denom)
        errors.append(worst)
    return errors
