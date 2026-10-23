# Model card — Pet breed classifier (EfficientNet-B0, LP-FT with frozen BatchNorm)

*Format after Mitchell et al. (2019), "Model cards for model reporting". Demo:
[live on Hugging Face Spaces](https://huggingface.co/spaces/Iansha/pet-breeds-gradcam) (ONNX in the browser, [code](../../demos/pets-web/); Gradio version in [`demos/pets-gradcam`](../../demos/pets-gradcam/)) · Report: [Lab C](../reports/lab-c-vision.md).*

## Model details

| | |
|---|---|
| Task | single-label image classification, 37 cat and dog breeds |
| Architecture | EfficientNet-B0 (`timm`), 4,054,945 parameters, input 160×160 RGB |
| Initialisation | ImageNet-1k weights from `timm`; the classification head starts from a converged logistic-regression probe on the frozen features |
| Training | LP-FT: 3 epochs of full fine-tuning, AdamW (lr 10⁻⁴, weight decay 0.05), cosine schedule, batch 32, label smoothing 0.1, random-resized crops + horizontal flips, **BatchNorm running statistics frozen** |
| Calibration | temperature scaling, T = 0.597, fitted on the validation split |
| Version | trained by `python -m lab_c.transfer` (C2), exported by `python -m lab_c.export`; weights as `model.safetensors` (16 MB) |
| Compute | CPU only (4-vCPU GitHub-hosted runner, ~3 min) |
| License | code MIT; weights derived from ImageNet-pretrained `timm` weights and the CC BY-SA 4.0 Oxford-IIIT Pet dataset |

## Intended use

- **Intended:** demonstrating transfer learning, calibration and attribution maps; education;
  a baseline for pet-breed classification research.
- **Out of scope:** identifying an individual animal; veterinary, insurance, breeding, adoption
  or legal decisions (e.g. breed-specific legislation — note that the model's weakest classes are
  exactly the bull-terrier breeds such laws target); any image that is not a photo of one cat or
  dog of the 37 breeds.

## Factors

Performance varies by breed (F1 from 0.65 to 0.99), with look-alike breeds, and with image
quality (noise, blur). Pose, lighting, background and photo source were not controlled: the
dataset is web photos of pets with a single dominant animal.

## Metrics and evaluation data

Oxford-IIIT Pet official test split (3,669 images), never used for training or selection.
Accuracy with a 95% percentile-bootstrap CI; expected calibration error (15 bins) and negative
log-likelihood; per-class F1; accuracy under synthetic corruptions.

## Quantitative analysis

| Metric | Value |
|---|---|
| Accuracy | **90.5%** [89.6, 91.4] |
| vs. linear probe on the same backbone | 90.4% — McNemar p = 0.61 (tied) |
| ECE (raw → calibrated) | 0.150 → 0.015 |
| NLL (raw → calibrated) | 0.452 → 0.313 |
| Hardest breeds (F1) | American pit bull terrier 0.65, Staffordshire bull terrier 0.70, ragdoll 0.76, American bulldog 0.77, birman 0.82 |
| Easiest breeds (F1) | Newfoundland, keeshond, Scottish terrier 0.99 |
| Most-confused pairs (test errors) | birman ↔ ragdoll 29, American pit bull ↔ Staffordshire bull terrier 26, British shorthair ↔ Russian blue 19 |
| Strongest corruption (severity 3) | dark 83.6%, low contrast 77.7%, Gaussian noise 49.9%, blur 48.5% |

Attribution maps were evaluated with a deletion test on 100 test images: Integrated Gradients are
faithful (deletion AUC 0.326 vs. 0.488 for random order, Holm p < 0.0001); Grad-CAM on the last
block (a 5×5 grid) is not (0.517, p = 0.33). The in-browser demo shows Grad-CAM on the final 5×5
feature map, one layer later, which was not tested separately.

## Training data

Oxford-IIIT Pet trainval split: 3,680 images, of which a stratified 10% (369) is held out for
validation and temperature fitting, leaving 3,311 for training. About 100 images per breed,
12 cat and 25 dog breeds. See the [data card](../data-cards.md#oxford-iiit-pet).

## Ethical considerations

- The model always answers with one of 37 breeds. A mixed-breed dog or an unrelated animal gets a
  confident-looking label; the calibrated probability is the only signal of doubt.
- The breed labels of the dataset are assigned from photos, not pedigree documents.
- Uploaded photos may contain people (owners often hold the pet). The hosted demo classifies them
  in the visitor's browser and never uploads them; the Gradio version processes them on whatever
  machine runs it.

## Caveats and recommendations

- Prefer the calibrated probabilities; treat predictions below ~50% as "unknown".
- Prefer Integrated Gradients over Grad-CAM to see what drives a prediction on this model.
- Re-evaluate before use on another photo source; the robustness tests are synthetic.
