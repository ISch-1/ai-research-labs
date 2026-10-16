# Image classification with little data and no GPU: what pre-training, fine-tuning, augmentation and attribution maps really buy

*Lab C technical report · [ai-research-labs](../../README.md) · code in [`lab-c-vision/`](../../lab-c-vision/) ·
detailed per-module reports in [`lab-c-vision/reports/`](../../lab-c-vision/reports/)*

## Abstract

We study image classification under two practical constraints: a few thousand labelled images
and CPU-only training. Starting from a perceptron and a backpropagation network written in
NumPy (verified against PyTorch to 10⁻¹⁵), we compare MLPs and CNNs, training from scratch and
transfer learning, augmentation recipes across data sizes, and post-hoc explanations. On
Oxford-IIIT Pets (37 breeds, 3.3k training images) a logistic regression on frozen ImageNet
features reaches **90.4%**, whereas a CNN trained from scratch reaches 24%. Full fine-tuning
initially *lost* to that linear probe; an ablation traced the loss to BatchNorm running
statistics drifting on small batches of augmented crops, and LP-FT with frozen BatchNorm gives
the best model (**90.5%** [89.6, 91.4]). The model is under-confident because of label
smoothing; temperature scaling reduces its expected calibration error tenfold. Finally, a
deletion test shows that the most plausible attribution maps (Grad-CAM) are not faithful on this
network, while Integrated Gradients are — and a planted-shortcut experiment shows that a model
with a perfect score on biased data can be useless on unbiased data.

## 1. Questions

1. **Fundamentals (C1).** Can backpropagation be implemented by hand and verified? Which
   training choices (activation, initialisation, optimiser, normalisation) actually matter?
2. **Transfer (C2).** How much do convolutional inductive bias and ImageNet pre-training buy
   with little data, and why can fine-tuning be worse than a frozen backbone?
3. **Augmentation (C3).** Which augmentation recipes help a small CNN, and does the answer
   depend on the amount of data?
4. **Errors (C4).** Which mistakes does the best model make, are its probabilities calibrated,
   and how robust is it to image corruptions?
5. **Interpretability (C5).** Are attribution maps faithful to what the model uses, and can
   they reveal shortcut learning?

## 2. Setup

| | |
|---|---|
| Data | Fashion-MNIST (60k/10k, 28 px), CIFAR-10 (50k/10k, 32 px), Oxford-IIIT Pets (3,311 train / 369 validation / 3,669 test, resized and centre-cropped to 160 px) — see the [data cards](../data-cards.md) |
| Models | NumPy MLP; PyTorch MLPs; a 0.29M-parameter VGG-style CNN; ImageNet backbones from `timm`: ResNet-18, EfficientNet-B0, MobileNetV3-Small, DeiT-Tiny |
| Training | one loop for every model: AdamW, cosine schedule, label smoothing 0.1 for fine-tuning; vectorised augmentation (`grid_sample`); trained models cached by (name, data, configuration) |
| Protocol | the test split is used once per model; the Pets validation split (10% of train, stratified) is used for model selection and temperature fitting |
| Statistics | 95% percentile-bootstrap CIs on test accuracy; exact McNemar tests against the best model; 3 seeds for ablations; Wilcoxon signed-rank with Holm correction for attribution methods |
| Compute | 4-core laptop CPU and 4-vCPU GitHub-hosted runners (long jobs run in parallel on CI, see [experiments.yml](../../.github/workflows/experiments.yml)) |

## 3. Results

### 3.1 Fundamentals (C1)

The perceptron converges on linearly separable data and never on XOR (42.5% training accuracy
after 100 epochs); one hidden layer of 8 tanh units solves XOR. The NumPy network
(784-128-64-10) passes a finite-difference gradient check (worst relative error 3.7·10⁻⁶) and
its gradients match PyTorch's autograd to 8.9·10⁻¹⁶; trained side by side from the same weights
on the same batches, the two implementations reach 86.14% and agree on 100% of test predictions.

Ablations of a 4-hidden-layer MLP on Fashion-MNIST (3 seeds, one factor at a time from ReLU,
default initialisation, Adam — 88.1%):

| Factor | Best | Worst | Not significantly different from the best |
|---|---|---|---|
| Normalisation | batch norm 89.7% | dropout 0.5 85.7% | — |
| Optimiser | SGD + Nesterov 88.4% | plain SGD 84.1% | AdamW (p = 0.07) |
| Initialisation | Xavier 88.8% | N(0, 0.01²) 86.7% | He (p = 0.34) |
| Activation | GELU 88.5% | sigmoid 85.5% | — |

### 3.2 Inductive bias and transfer learning (C2)

With the same budget (15 epochs) a 0.29M-parameter CNN reaches **85.4%** [84.7, 86.1] on
CIFAR-10 and a 1.7M-parameter MLP 55.4% [54.5, 56.3] (McNemar p < 0.0001).

On Pets (selected rows; all 14 in the [C2 report](../../lab-c-vision/reports/c2_transfer.md)):

| Method | Trainable params | Test accuracy (95% CI) |
|---|---|---|
| CNN from scratch (20 epochs, 112 px) | 0.29M | 24.3% [22.8, 25.7] |
| MobileNetV3-Small, linear probe | 38k | 81.9% [80.6, 83.1] |
| ResNet-18, linear probe | 19k | 85.9% [84.8, 87.0] |
| DeiT-Tiny, linear probe | 7k | 86.9% [85.8, 88.0] |
| EfficientNet-B0, linear probe | 47k | 90.4% [89.3, 91.3] |
| EfficientNet-B0, fine-tuned (random head) | 4.05M | 80.4% [79.1, 81.7] |
| EfficientNet-B0, LP-FT | 4.05M | 84.3% [83.2, 85.4] |
| **EfficientNet-B0, LP-FT, frozen BatchNorm** | 4.05M | **90.5% [89.6, 91.4]** |

Frozen ImageNet features separate the breeds by themselves: 5-nearest-neighbour accuracy is
88.5% on EfficientNet-B0 features and 22.7% on the scratch CNN's.

**Why fine-tuning lost.** One epoch of fine-tuning from the probe's head (EfficientNet-B0),
changing one thing at a time:

| Variant | Validation | Test |
|---|---|---|
| linear probe, no fine-tuning | 92.4% | 90.4% |
| LP-FT as configured (lr 10⁻⁴) | 84.6% | 83.2% |
| learning rate ÷ 10 | 73.7% | 68.7% |
| no random crops | 86.2% | 85.4% |
| no label smoothing | 85.9% | 84.7% |
| **BatchNorm statistics frozen** | **91.1%** | **87.8%** |

A lower learning rate makes things *worse*, which rules out "the weights moved too far". What
changes are the BatchNorm running statistics, re-estimated from batches of 32 augmented crops
of one small dataset; freezing them recovers most of the gap. With frozen statistics, LP-FT
improves every backbone (ResNet-18 78.7% → 85.6%, MobileNetV3 71.1% → 79.5%) and EfficientNet-B0
ties its linear probe (McNemar p = 0.61), 10 points above naive fine-tuning.

### 3.3 Augmentation × data size (C3)

SmallCNN on CIFAR-10 with 10%, 25% and 100% of the training images (epochs scaled up for the
smaller fractions):

| Recipe | 10% | 25% | 100% |
|---|---|---|---|
| none | 68.0% | 76.5% | 84.8% |
| **flip + crop** | **72.0%** (+3.9) | **79.5%** (+3.0) | **85.8%** (+1.0) |
| + colour jitter | 70.1% | 79.2% | 85.3% |
| + cutout | 70.0% | 78.2% | 84.6% |
| + mixup | 70.1% | 77.1% | 83.1% |
| + CutMix | 66.3% | 74.0% | 80.4% |

Flip + crop is the best recipe at every size and its gain grows as data shrink (each of its
gains has McNemar p < 0.01). Without augmentation the train-test gap reaches 32 points at 10% of the data
(11 with flip + crop). Mixup and CutMix shrink the gap the most but over-regularise a model this
small at this budget.

### 3.4 Errors, calibration and robustness (C4)

The best model's errors are between look-alike breeds: birman ↔ ragdoll (29 test confusions),
American pit bull ↔ Staffordshire bull terrier (26), British shorthair ↔ Russian blue (19). The
model is under-confident — label smoothing pushes probabilities toward uniform — and one
temperature fitted on the validation split fixes it without changing any prediction:

| Model | T | ECE (15 bins) | NLL |
|---|---|---|---|
| EfficientNet-B0, LP-FT, frozen BN | 0.150 → 0.015 | 0.452 → 0.313 |
| CNN from scratch | 0.086 → 0.025 | 2.721 → 2.658 |

Under the strongest synthetic corruption the pre-trained model keeps 84% accuracy in the dark,
78% at low contrast, but only ~49% under Gaussian noise or blur; on average it keeps 72% of its
clean accuracy, the scratch CNN 19%.

### 3.5 Faithfulness of attribution maps and shortcut learning (C5)

Deletion test on 100 test images: the pixels a map ranks highest are replaced by the mean
colour (0–50% of the image) and the predicted class's probability is tracked, relative to the
clean image. Lower area = the map found pixels the model relies on.

| Method | Deletion AUC | Holm p vs. random order |
|---|---|---|
| Integrated Gradients | **0.326** | < 0.0001 |
| Gradient saliency | 0.328 | < 0.0001 |
| random order | 0.488 | — |
| Grad-CAM | 0.517 | 0.33 |

Grad-CAM produces the most convincing maps — always on the animal's face — but on EfficientNet-B0
at 160 px it is computed on a 5×5 grid, too coarse to rank pixels better than chance.

**Shortcut learning.** A CNN trained on CIFAR-10 where a 5×5 coloured corner patch encodes the
class in 95% of images scores 100% when the patch is right, 3.7% when it lies (it follows the
patch 95.6% of the time) and 64.7% without a patch; the same CNN trained on clean images stays at
~85% in every condition. Grad-CAM exposes it: 10.5% of the shortcut model's attention falls on a
patch covering 2.4% of the image (0.7% for the clean model).

## 4. Discussion

- **With little data, use the features before tuning the weights.** A linear probe on a good
  backbone was within noise of the best fine-tuned model, trains in under a minute on a CPU and
  is far less fragile. Full fine-tuning paid off only after an ablation found the real cause of
  its failure.
- **Debug training with one-factor ablations, not by tuning harder.** The learning-rate
  ablation pointed *away* from the intuitive explanation; the BatchNorm ablation found it.
- **Report calibration next to accuracy.** The same regularisation that helps accuracy (label
  smoothing) made raw probabilities misleading; a single scalar fixed them. The demo shows
  calibrated probabilities for this reason.
- **Test explanations before trusting them.** Plausibility is not faithfulness: the map a human
  finds most convincing failed the deletion test here. Grad-CAM was still useful for a coarse
  question (does the model look at the corner patch?).

## 5. Limitations

- One dataset per question and one seed for the large models; CIs capture test-set sampling,
  not training variance (the C1 ablations use 3 seeds).
- CPU budgets: 3–5 fine-tuning epochs and 15–40 CNN epochs. Rankings could change with longer
  schedules (mixup and CutMix in particular usually need more epochs to pay off).
- The deletion test replaces pixels with the mean colour, which creates out-of-distribution
  images; insertion tests or ROAR (retrain after removal) would strengthen the conclusion.
- Corruptions are synthetic (noise, blur, contrast, darkness), not natural distribution shift;
  the Pets model is not evaluated on photos from other sources.
- Pets is a curated dataset of pure-bred animals: the model always outputs one of 37 breeds,
  including for mixed breeds or other animals (see the [model card](../model-cards/pets-efficientnet-b0.md)).

## 6. Reproducibility

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,lab-c]"
python -m lab_c.fundamentals --config lab-c-vision/configs/c1_fundamentals.yaml
python -m lab_c.transfer     --config lab-c-vision/configs/c2_transfer.yaml
python -m lab_c.errors       --config lab-c-vision/configs/c4_errors.yaml
python -m lab_c.interpret    --config lab-c-vision/configs/c5_interpret.yaml
gh workflow run experiments.yml -f experiment=c3_augmentation   # C3 on GitHub Actions
```

Every number above comes from the reports these commands regenerate; fold- and model-level
results are in `lab-c-vision/reports/results/*.csv`.

## References

- Kumar, Raghunathan, Jones, Ma, Liang (2022). *Fine-tuning can distort pretrained features and underperform out-of-distribution.* ICLR.
- Ioffe, Szegedy (2015). *Batch normalization.* ICML.
- Guo, Pleiss, Sun, Weinberger (2017). *On calibration of modern neural networks.* ICML.
- Müller, Kornblith, Hinton (2019). *When does label smoothing help?* NeurIPS.
- Hendrycks, Dietterich (2019). *Benchmarking neural network robustness to common corruptions and perturbations.* ICLR.
- Sundararajan, Taly, Yan (2017). *Axiomatic attribution for deep networks.* ICML.
- Selvaraju et al. (2017). *Grad-CAM: visual explanations from deep networks via gradient-based localization.* ICCV.
- Samek, Binder, Montavon, Lapuschkin, Müller (2017). *Evaluating the visualization of what a deep neural network has learned.* IEEE TNNLS.
- Adebayo et al. (2018). *Sanity checks for saliency maps.* NeurIPS.
- Geirhos et al. (2020). *Shortcut learning in deep neural networks.* Nature Machine Intelligence.
- Zhang, Cissé, Dauphin, Lopez-Paz (2018). *mixup.* ICLR · Yun et al. (2019). *CutMix.* ICCV · DeVries, Taylor (2017). *Cutout.* arXiv.
- Tan, Le (2019). *EfficientNet.* ICML · He et al. (2016). *ResNet.* CVPR · Howard et al. (2019). *MobileNetV3.* ICCV · Touvron et al. (2021). *DeiT.* ICML.
- Parkhi, Vedaldi, Zisserman, Jawahar (2012). *Cats and dogs.* CVPR · Krizhevsky (2009). *CIFAR-10* · Xiao, Rasul, Vollgraf (2017). *Fashion-MNIST.* arXiv.
