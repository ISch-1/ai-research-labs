# Lab C — Computer Vision

> Image classification from first principles to transfer learning: a perceptron and a
> backpropagation network written in NumPy and checked against PyTorch, CNNs vs. MLPs, ImageNet
> backbones as frozen features or fine-tuned, data augmentation under scarce data, error analysis
> and calibration, and attribution maps whose faithfulness is measured rather than assumed.

## Key findings

**C1 — Fundamentals.**
- The perceptron converges on linearly separable data and never on XOR; one hidden layer solves it.
- A NumPy network with hand-written backpropagation passes a finite-difference gradient check
  (worst relative error ~10⁻⁶) and matches PyTorch's autograd to ~10⁻¹⁵; trained side by side from
  the same weights on the same batches, both predict the same class for 100% of Fashion-MNIST
  test images.
- In a 4-hidden-layer MLP (baseline: ReLU, default init, Adam — 88.1%), batch normalisation is the
  largest single gain (89.7%), plain SGD the largest loss (84.1%), and a sigmoid activation or a
  too-small initialisation cost 2.7 and 1.4 points. Most gaps to each group's best are significant
  (McNemar); He vs. Xavier initialisation (p = 0.34) and AdamW vs. Nesterov SGD (p = 0.07) are not.

**C2 — CNNs and transfer learning.**
- Inductive bias: with the same training budget a 0.29M-parameter CNN reaches 85.4% on CIFAR-10,
  a 1.7M-parameter MLP 55.4%.
- On Oxford-IIIT Pets (37 breeds, 3.3k training images) a CNN trained from scratch reaches 24%,
  while a logistic regression on frozen ImageNet features reaches 82–90% in under a minute
  (EfficientNet-B0 best, 90.4%). The frozen features separate breeds by themselves (5-NN accuracy
  88.5% vs. 22.7% for the scratch CNN's features).
- Fine-tuning the whole network on CPU (3–5 epochs) *loses* to that linear probe. Starting from the
  probe's head (LP-FT) is better than a random head (84.3% vs. 80.4% for EfficientNet) but still
  below the probe. A one-epoch ablation points at **BatchNorm**: a lower learning rate makes things
  worse (68.5%), while freezing the BatchNorm statistics recovers most of the gap (87.8%) — the
  running statistics drift on small augmented batches.

<!-- C2-BN -->

**C3 — Data augmentation.**
- Flip + crop is the best recipe at every data size, and its gain grows as data shrink:
  +1.0 point with all 50k CIFAR-10 images, +3.0 with 25%, +3.9 with 10%.
- Without augmentation the generalisation gap (training minus test accuracy) reaches 32 points at
  10% of the data, vs. 11 with flip + crop.
- Mixup and CutMix hurt this small CNN at this budget (CutMix −4.4 points with all data): they shrink
  the gap the most, but over-regularise.

<!-- C4-C5 -->

## Modules

| Module | Question | Code | Report |
|---|---|---|---|
| C1 | Perceptron, backprop by hand, and which training choices matter | [`fundamentals.py`](src/lab_c/fundamentals.py), [`scratch.py`](src/lab_c/scratch.py) | [C1](reports/c1_fundamentals.md) |
| C2 | CNN vs. MLP; scratch vs. linear probe vs. fine-tuning (naive, LP-FT, frozen BN) | [`transfer.py`](src/lab_c/transfer.py) | [C2](reports/c2_transfer.md) |
| C3 | Augmentation recipes × data size | [`augmentation.py`](src/lab_c/augmentation.py), [`augment.py`](src/lab_c/augment.py) | [C3](reports/c3_augmentation.md) |
| C4 | Confusions, confident errors, calibration, robustness to corruptions | [`errors.py`](src/lab_c/errors.py) | [C4](reports/c4_errors.md) |
| C5 | Saliency, Integrated Gradients, Grad-CAM; deletion-test faithfulness; shortcut learning | [`interpret.py`](src/lab_c/interpret.py) | [C5](reports/c5_interpret.md) |

## Methodology

- **Data.** Fashion-MNIST, CIFAR-10 and Oxford-IIIT Pets from the Hugging Face Hub, decoded once to
  uint8 tensors. Pets images are resized to 160 px; a stratified 10% of its training set is held
  out for model selection, early decisions and temperature scaling.
- **Training.** One loop for every model ([`train.py`](src/lab_c/train.py)): AdamW, cosine
  schedule, optional label smoothing, LP-FT and frozen-BatchNorm options; trained models are cached
  by (name, data, configuration) so experiments resume and later modules reuse them.
- **Augmentation** is vectorised on batches with one `grid_sample` for flips, crops and
  random-resized crops, plus colour jitter, cutout, mixup and cutmix.
- **Statistics.** Bootstrap 95% CIs on test accuracy, McNemar against the best model, several seeds
  for ablations, Wilcoxon + Holm for the attribution methods.
- **Attribution methods** are implemented from their papers (gradients, Riemann-sum Integrated
  Gradients, Grad-CAM through a tensor hook) and tested — e.g. Integrated Gradients is exact on a
  linear model.

## Run

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,lab-c]"

python -m lab_c.fundamentals --config lab-c-vision/configs/c1_fundamentals.yaml   # ~10 min
python -m lab_c.transfer     --config lab-c-vision/configs/c2_transfer.yaml       # ~3 h on CPU
python -m lab_c.errors       --config lab-c-vision/configs/c4_errors.yaml         # reuses C2's models
python -m lab_c.interpret    --config lab-c-vision/configs/c5_interpret.yaml
# long runs on GitHub Actions instead (parallel jobs):
gh workflow run experiments.yml -f experiment=c3_augmentation
```

Every config has an offline `*_smoke.yaml` twin that CI runs. Reports state the hardware their
timings come from.
