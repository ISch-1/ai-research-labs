---
title: Pet Breeds + Attribution Maps
emoji: 🐾
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: 37 cat & dog breeds, calibrated, with Grad-CAM and IG maps
---

# Pet breed classifier with attribution maps

A Gradio demo of the best model from **Lab C** of
[ai-research-labs](https://github.com/ISch-1/ai-research-labs): it names one of the 37 breeds of
the Oxford-IIIT Pet dataset, gives **calibrated** probabilities, and shows which pixels support
its answer with **Grad-CAM** or **Integrated Gradients**.

![Demo output on four test images](https://raw.githubusercontent.com/ISch-1/ai-research-labs/main/docs/assets/pets-demo.png)

| | |
|---|---|
| Model | EfficientNet-B0 (ImageNet weights from `timm`), LP-FT with frozen BatchNorm — 4.05M parameters |
| Test accuracy | **90.5%** [89.6, 91.4] on 3,669 held-out images (95% bootstrap CI) |
| Calibration | temperature T = 0.60 fitted on the validation split; ECE 0.150 → 0.015 |
| Input | resized and centre-cropped to 160×160 px, as in training |
| Speed (laptop CPU) | ~0.1 s for a prediction + Grad-CAM, ~1.2 s with Integrated Gradients (32 steps) |

Model card: [docs/model-cards/pets-efficientnet-b0.md](https://github.com/ISch-1/ai-research-labs/blob/main/docs/model-cards/pets-efficientnet-b0.md).

## Read the maps with care

Lab C measured whether these maps are **faithful** — whether deleting the pixels a map ranks
highest makes the prediction fall faster than deleting random pixels:

- **Integrated Gradients** passes (deletion AUC 0.326 vs. 0.488 for random order, Holm p < 0.0001).
- **Grad-CAM** looks the most convincing but does **not** pass on this network (0.517 vs. 0.488,
  p = 0.33): on EfficientNet-B0 at 160 px it is computed on a 5×5 grid, too coarse to single out
  the pixels the model uses.

## Limitations

- Only the 37 Oxford-IIIT breeds: any other animal, a mixed breed or an object still gets a
  breed name. Low probabilities (the app says "unsure" below 50%) are the only warning.
- Look-alike pairs are the main errors: birman ↔ ragdoll, American pit bull ↔ Staffordshire bull
  terrier, British shorthair ↔ Russian blue. The last example is one of them.
- Accuracy roughly halves under strong noise or blur (~49%); darkness and low contrast cost less.
- Trained on 3.3k photos of pets from the web: little variety in pose, lighting and breed
  standard, and no evaluation on photos from other sources.

## Run locally

```bash
pip install gradio -r requirements.txt
python app.py        # http://127.0.0.1:7860
```

The weights (`model.safetensors`, 16 MB) are not in git; recreate them from the repository root
with `python -m lab_c.export --config lab-c-vision/configs/export_demo.yaml` (reuses the model
trained in C2) or download them from the Space.

Example images: Oxford-IIIT Pet dataset (O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,
*Cats and Dogs*, CVPR 2012), CC BY-SA 4.0.
