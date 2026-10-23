---
title: Pet Breeds + Class-Activation Maps
emoji: 🐾
colorFrom: indigo
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: 37 cat & dog breeds, calibrated, in your browser (ONNX)
---

# Pet breed classifier — in the browser

**Live demo:** <https://huggingface.co/spaces/Iansha/pet-breeds-gradcam>

The best model from **Lab C** of [ai-research-labs](https://github.com/ISch-1/ai-research-labs)
running entirely in the visitor's browser with [ONNX Runtime Web](https://onnxruntime.ai/): no
server, and photos never leave the device. It names one of the 37 breeds of the Oxford-IIIT Pet
dataset, gives **calibrated** probabilities and draws a **class-activation map**.

| | |
|---|---|
| Model | EfficientNet-B0 (ImageNet weights from `timm`), LP-FT with frozen BatchNorm, exported to ONNX (opset 17, 15 MB) |
| Test accuracy | **90.5%** [89.6, 91.4] on 3,669 held-out images (95% bootstrap CI) |
| Calibration | temperature T = 0.60 fitted on the validation split; ECE 0.150 → 0.015 |
| Input | centre-cropped to a square and resized to 160×160 px, as in training |
| Agreement with PyTorch | checked at export with ONNX Runtime: same top-1 class, logits within 10⁻⁵ |

## How the map is computed without gradients

The ONNX graph has two outputs: the logits and, for every breed *k*, the map
Σ<sub>c</sub> W<sub>kc</sub> F<sub>c</sub> over the final 5×5 feature map *F*. Because the head is
global average pooling followed by one linear layer, the gradient of the logit with respect to
*F* is W<sub>kc</sub>/25 at every position, so Grad-CAM at that layer is exactly ReLU of this map divided by 25
(a test in the repository checks it against autograd). The page keeps the top breed's map,
upsamples it bilinearly and blends it over the crop.

Lab C's deletion test found that Grad-CAM on the network's last block — also a 5×5 grid, one layer
earlier — does not single out the pixels the model relies on better than chance, while Integrated
Gradients do; treat this map as a coarse pointer to *where* the evidence is. Integrated Gradients need
gradients, so they are only in the [Python/Gradio version](https://github.com/ISch-1/ai-research-labs/tree/main/demos/pets-gradcam).

## Limitations

- Only the 37 Oxford-IIIT breeds: other animals, mixed breeds and objects still get a breed name;
  below 50% the page says it is unsure.
- Look-alike pairs are the main errors (birman ↔ ragdoll, American pit bull ↔ Staffordshire bull
  terrier); the last example image is one of them.
- Accuracy roughly halves under strong noise or blur.
- Model card: [docs/model-cards/pets-efficientnet-b0.md](https://github.com/ISch-1/ai-research-labs/blob/main/docs/model-cards/pets-efficientnet-b0.md).

## Build

From the repository root (the ONNX file, `config.json` and the examples are generated, not in git):

```bash
python -m lab_c.export --config lab-c-vision/configs/export_demo.yaml
pip install -e ".[demo]" && python demos/pets-web/export_onnx.py
python -m http.server 8000 --directory demos/pets-web     # http://localhost:8000
```

Example images: Oxford-IIIT Pet dataset (Parkhi et al., *Cats and Dogs*, CVPR 2012), CC BY-SA 4.0.
