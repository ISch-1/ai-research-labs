"""Unit tests for Lab C building blocks (offline)."""

import numpy as np
import pytest
import torch
from torch import nn

from lab_c import augment as A
from lab_c.datasets import ImageSet, synthetic_shapes
from lab_c.errors import corrupt, ece, fit_temperature
from lab_c.interpret import add_patch, deletion_curve, grad_cam, integrated_gradients, saliency
from lab_c.nets import MLP, SmallCNN, gradcam_layer
from lab_c.scratch import NumpyMLP, Perceptron, gradient_check
from lab_c.train import TrainConfig, fit, predict, train_or_load

# ------------------------------------------------------------------------- scratch


def test_perceptron_separable_vs_xor():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    assert Perceptron().fit(X, y).converged
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    assert not Perceptron(max_epochs=50).fit(X_xor, np.array([0, 1, 1, 0])).converged


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_numpy_mlp_gradients_are_correct(activation):
    rng = np.random.default_rng(1)
    model = NumpyMLP([6, 5, 4, 3], activation, seed=1)
    errors = gradient_check(model, rng.normal(size=(16, 6)), rng.integers(0, 3, 16))
    assert max(errors) < 1e-5


def test_numpy_mlp_matches_torch_autograd():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(10, 8)), rng.integers(0, 4, 10)
    model = NumpyMLP([8, 7, 4], "relu", seed=2)
    loss, grads = model.loss_and_grads(X, y)
    net = model.to_torch()
    torch_loss = nn.functional.cross_entropy(net(torch.from_numpy(X)), torch.from_numpy(y))
    torch_loss.backward()
    assert loss == pytest.approx(torch_loss.item(), abs=1e-12)
    for g, p in zip(grads, net.parameters(), strict=True):
        np.testing.assert_allclose(
            g, p.grad.numpy().T if g.ndim == 2 else p.grad.numpy(), atol=1e-12
        )


def test_numpy_mlp_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    model = NumpyMLP([2, 8, 2], "tanh", seed=1)
    for epoch in range(500):
        model.fit_epoch(X, y, lr=0.1, batch_size=4, seed=epoch)
    assert model.predict(X).tolist() == [0, 1, 1, 0]


# ------------------------------------------------------------------------ datasets


def test_imageset_fraction_and_split():
    data = synthetic_shapes(300)
    part = data.stratified_fraction(0.2, seed=0)
    assert len(part) == 60
    assert set(part.labels.tolist()) == {0, 1, 2}
    train, val = data.split(0.1, seed=0)
    assert len(train) + len(val) == 300
    assert not set(map(tuple, val.images.reshape(len(val), -1)[:, :8].tolist())) & set(
        map(tuple, train.images.reshape(len(train), -1)[:, :8].tolist())
    )


# ------------------------------------------------------------------------- augment


@pytest.mark.parametrize(
    "spec", [None, "flip+crop", "resized+flip", "color", "cutout", "flip+mixup", "crop+cutmix"]
)
def test_augmentations_keep_shape_and_range(spec):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(6, 3, 16, 16)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    xa, target = A.augment(x, y, spec, 3, g)
    assert xa.shape == x.shape
    assert xa.min() >= 0 and xa.max() <= 1
    torch.testing.assert_close(target.sum(1), torch.ones(6))


def test_augment_rejects_unknown_and_conflicting_ops():
    with pytest.raises(ValueError):
        A.parse("flip+rotate")
    with pytest.raises(ValueError):
        A.parse("mixup+cutmix")


# --------------------------------------------------------------------------- train


def test_cnn_beats_mlp_on_shapes():
    train, test = synthetic_shapes(600, seed=0), synthetic_shapes(300, seed=1)
    cfg = TrainConfig(epochs=6, lr=3e-3, batch_size=64)
    torch.manual_seed(0)
    cnn = SmallCNN(3, 3, width=16)
    fit(cnn, train, cfg)
    mlp = MLP(3 * 32 * 32, [64], 3)
    fit(mlp, train, cfg)
    acc = lambda m: float((predict(m, test).argmax(1) == test.labels).float().mean())  # noqa: E731
    assert acc(cnn) > 0.9 > acc(mlp)


def test_train_or_load_reuses_the_cached_model(monkeypatch):
    data = synthetic_shapes(120)
    cfg = TrainConfig(epochs=1, batch_size=32)
    build = lambda: SmallCNN(3, 3, width=4)  # noqa: E731
    first, _ = train_or_load("cache-test", build, data, cfg)

    def boom(*args, **kwargs):
        raise AssertionError("should not retrain")

    monkeypatch.setattr("lab_c.train.fit", boom)
    again, history = train_or_load("cache-test", build, data, cfg)
    for a, b in zip(first.state_dict().values(), again.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b)
    assert history.epoch_seconds


# ------------------------------------------------------------------------- errors


def test_ece_is_zero_for_a_calibrated_model_and_positive_otherwise():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 20000)
    correct = rng.random(20000) < conf  # correct exactly as often as claimed
    labels = np.zeros(20000, int)
    pred = np.where(correct, 0, 1)
    probs = np.zeros((20000, 2))
    probs[np.arange(20000), pred] = conf
    probs[np.arange(20000), 1 - pred] = 1 - conf
    assert ece(probs, labels) < 0.02
    overconfident = np.where(probs.max(1, keepdims=True) == probs, 0.999, 0.001)
    assert ece(overconfident, labels) > 0.1


def test_temperature_scaling_recovers_the_true_temperature():
    torch.manual_seed(0)
    true_logits = torch.randn(4000, 5) * 2
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    assert fit_temperature(true_logits * 3, labels) == pytest.approx(3.0, rel=0.15)


@pytest.mark.parametrize("kind", ["gaussian_noise", "blur", "contrast", "dark"])
def test_corruptions(kind):
    images = synthetic_shapes(8).images
    mild, strong = corrupt(images, kind, 1), corrupt(images, kind, 3)
    assert mild.dtype == torch.uint8 and mild.shape == images.shape
    diff = lambda x: (x.float() - images.float()).abs().mean()  # noqa: E731
    assert 0 < diff(mild) < diff(strong)


# ---------------------------------------------------------------------- interpret


def test_integrated_gradients_is_exact_for_a_linear_model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    x = torch.randn(2, 3, 4, 4)
    target = torch.tensor([0, 1])
    ig = integrated_gradients(model, x, target, torch.zeros_like(x), steps=4)
    w = model[1].weight.detach().view(2, 3, 4, 4)
    expected = torch.stack([(x[i] * w[target[i]]).abs().sum(0) for i in range(2)])
    torch.testing.assert_close(ig, expected)


def test_saliency_and_gradcam_shapes():
    model = SmallCNN(3, 3, width=4).eval()
    x = torch.randn(2, 3, 16, 16)
    target = torch.tensor([0, 2])
    assert saliency(model, x, target).shape == (2, 16, 16)
    cam = grad_cam(model, x, target, gradcam_layer(model))
    assert cam.shape == (2, 16, 16) and (cam >= 0).all()


def test_deletion_curve_starts_at_the_clean_probability():
    model = SmallCNN(3, 3, width=4).eval()
    x = torch.randn(3, 3, 8, 8)
    target = torch.tensor([0, 1, 2])
    curve = deletion_curve(model, x, target, torch.rand(3, 8, 8), steps=4)
    with torch.no_grad():
        clean = model(x).softmax(1)[torch.arange(3), target].numpy()
    np.testing.assert_allclose(curve[:, 0], clean, rtol=1e-5)
    assert curve.shape == (3, 5)


def test_a_model_trained_with_the_shortcut_follows_the_patch():
    """Pure-noise images: the patch is the only signal, so the model must learn it."""
    g = torch.Generator().manual_seed(0)
    noise = lambda n: ImageSet("noise", torch.randint(0, 256, (n, 3, 16, 16), generator=g, dtype=torch.uint8),  # noqa: E731
                                torch.randint(0, 4, (n,), generator=g), list("abcd"), "noise")  # fmt: skip
    train, test = noise(800), noise(200)
    patched = add_patch(train, train.labels, 1.0, 4, 0, "noise_patched")
    model = SmallCNN(3, 4, width=8)
    fit(model, patched, TrainConfig(epochs=6, lr=3e-3, batch_size=64))
    wrong = (test.labels + 1) % 4
    lying = add_patch(test, wrong, 1.0, 4, 0, "noise")
    follows = float((predict(model, lying).argmax(1) == wrong).float().mean())
    assert follows > 0.9


def test_lpft_head_only_stage_leaves_the_backbone_untouched():
    data = synthetic_shapes(120)
    model = SmallCNN(3, 3, width=4)
    before = {
        k: v.clone()
        for k, v in model.stages.state_dict().items()
        if "running" not in k and "num_batches" not in k
    }
    fit(model, data, TrainConfig(epochs=1, head_only_epochs=1, batch_size=32))
    after = model.stages.state_dict()
    for k, v in before.items():
        torch.testing.assert_close(after[k], v)  # frozen during the head-only stage
    history = fit(model, data, TrainConfig(epochs=2, head_only_epochs=1, batch_size=32))
    assert len(history.train_loss) == 2
    assert all(p.requires_grad for p in model.parameters())  # unfrozen afterwards
