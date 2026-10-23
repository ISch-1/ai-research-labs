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
- Fine-tuning the whole network on CPU (3–5 epochs) at first *lost* to that linear probe: 80.4% with
  a random head, 84.3% starting from the probe's head (LP-FT). A one-epoch ablation pointed at
  **BatchNorm**, not the weights: a 10× lower learning rate made things worse (68.7%), while
  freezing the BatchNorm statistics recovered most of the gap (87.8%) — the running statistics
  drift on small batches of augmented crops.
- With that fix, LP-FT with frozen BatchNorm is the best Pets model: EfficientNet-B0 90.5%, tied
  with its linear probe (McNemar p = 0.61) and +10 points over naive fine-tuning; freezing BatchNorm
  also lifts LP-FT from 78.7% to 85.6% (ResNet-18) and from 71.1% to 79.5% (MobileNetV3).

**C3 — Data augmentation.**
- Flip + crop is the best recipe at every data size, and its gain grows as data shrink:
  +1.0 point with all 50k CIFAR-10 images, +3.0 with 25%, +3.9 with 10%.
- Without augmentation the generalisation gap (training minus test accuracy) reaches 32 points at
  10% of the data, vs. 11 with flip + crop.
- Mixup and CutMix hurt this small CNN at this budget (CutMix −4.4 points with all data): they shrink
  the gap the most, but over-regularise.

**C4 — Errors, calibration, robustness** (EfficientNet-B0, LP-FT with frozen BatchNorm, 90.5%).
- The errors are the ones a person would make: birman ↔ ragdoll (29 confusions), American pit bull ↔
  Staffordshire bull terrier (26), British shorthair ↔ Russian blue (19); the two bull-terrier
  breeds are the hardest classes (F1 0.65 and 0.70).
- The model is *under*-confident — the price of label smoothing: temperature scaling with T = 0.60
  (fitted on the validation split) cuts the expected calibration error from 0.150 to 0.015 and the
  NLL from 0.452 to 0.313 without changing a single prediction.
- It shrugs off low contrast and darkness (78–84% at the strongest severity) but halves under strong
  noise or blur (~49%); on average it keeps 72% of its clean accuracy vs. 19% for the CNN trained
  from scratch.

**C5 — Interpretability.**
- Faithfulness is not the same as plausibility. Grad-CAM produces the most convincing maps (always
  the animal's face) but, deleting pixels in its order, the prediction falls no faster than with a
  random order (deletion AUC 0.517 vs. 0.488, p = 0.33): on EfficientNet it works on a 5×5 grid,
  too coarse for a pixel-level test. Integrated Gradients (0.326) and plain gradients (0.328) find
  pixels the model really uses (Holm p < 0.0001 vs. random).
- Shortcut learning: a CNN trained on CIFAR-10 where a 5×5 coloured corner patch reveals the class
  in 95% of images scores 100% when the patch tells the truth, 3.7% when it lies (it follows the
  patch 95.6% of the time) and 64.7% without it, vs. a steady ~85% for the same CNN trained on clean
  images. Grad-CAM exposes it: 10.5% of its attention falls on a patch covering 2.4% of the image,
  vs. 0.7% for the clean model.

The full write-up — questions, setup, results with CIs and limitations — is the
[Lab C technical report](../docs/reports/lab-c-vision.md).

## Modules

| Module | Question | Code | Report |
|---|---|---|---|
| C1 | Perceptron, backprop by hand, and which training choices matter | [`fundamentals.py`](src/lab_c/fundamentals.py), [`scratch.py`](src/lab_c/scratch.py) | [C1](reports/c1_fundamentals.md) |
| C2 | CNN vs. MLP; scratch vs. linear probe vs. fine-tuning (naive, LP-FT, frozen BN) | [`transfer.py`](src/lab_c/transfer.py) | [C2](reports/c2_transfer.md) |
| C3 | Augmentation recipes × data size | [`augmentation.py`](src/lab_c/augmentation.py), [`augment.py`](src/lab_c/augment.py) | [C3](reports/c3_augmentation.md) |
| C4 | Confusions, confident errors, calibration, robustness to corruptions | [`errors.py`](src/lab_c/errors.py) | [C4](reports/c4_errors.md) |
| C5 | Saliency, Integrated Gradients, Grad-CAM; deletion-test faithfulness; shortcut learning | [`interpret.py`](src/lab_c/interpret.py) | [C5](reports/c5_interpret.md) |
| Demo | The best Pets model with calibrated probabilities and attribution maps (Gradio) | [`export.py`](src/lab_c/export.py), [`demos/pets-gradcam`](../demos/pets-gradcam/) | [live demo](https://huggingface.co/spaces/Iansha/pet-breeds-gradcam) · [model card](../docs/model-cards/pets-efficientnet-b0.md) |

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
python -m lab_c.export       --config lab-c-vision/configs/export_demo.yaml      # weights for the demo
# long runs on GitHub Actions instead (parallel jobs):
gh workflow run experiments.yml -f experiment=c3_augmentation
```

Every config has an offline `*_smoke.yaml` twin that CI runs. Reports state the hardware their
timings come from.
