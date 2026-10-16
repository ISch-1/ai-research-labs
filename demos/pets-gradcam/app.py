"""Gradio demo: 37 cat and dog breeds, calibrated probabilities and an attribution map.

Run with ``python app.py`` and open http://127.0.0.1:7860.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from inference import HERE, PetClassifier, overlay
from PIL import Image

clf = PetClassifier()
cfg = clf.config

METHODS = {
    "Grad-CAM": "coarse (a 5×5 grid on this network) and easy to read",
    "Integrated Gradients": "pixel-level; in Lab C's deletion test it tracked what the model uses, Grad-CAM did not",
}


def classify(
    image: Image.Image | None, method: str
) -> tuple[dict[str, float], Image.Image | None, str]:
    if image is None:
        return {}, None, ""
    crop, x = clf.preprocess(image)
    probs = clf.probabilities(x)
    top = int(probs.argmax())
    if method == "Grad-CAM":
        heat = clf.grad_cam(x, top)
        shown = overlay(crop, heat)
    else:
        heat = clf.integrated_gradients(x, top)
        shown = overlay(crop, heat, smooth=5)
    shown = shown.resize((2 * clf.size, 2 * clf.size), Image.Resampling.BICUBIC)
    p = float(probs[top])
    note = (
        f"**{clf.classes[top]}** with {p:.0%} probability."
        if p >= 0.5
        else f"Unsure: the best guess, **{clf.classes[top]}**, has only {p:.0%}."
    )
    note += f" The map shows the evidence for {clf.classes[top]} ({method}: {METHODS[method]})."
    return {c: float(v) for c, v in zip(clf.classes, probs, strict=True)}, shown, note


def _hard_last(path: Path) -> tuple[bool, str]:
    return path.stem.endswith("_hard"), path.stem


DESCRIPTION = f"""
Upload a photo of a cat or a dog. The model — **{cfg["architecture"]}**, ImageNet-pretrained,
then {cfg["training"]} — names one of 37 breeds and shows which pixels support its answer.
It is right on **{cfg["test_accuracy"]:.1%}** of {cfg["test_images"]:,} held-out test images.
Probabilities are **temperature-scaled** (T = {cfg["temperature"]:.2f}, fitted on a validation
split), so "70%" means right about 70% of the time.

It only knows these 37 breeds: any other animal, mixed breed or object still gets a breed name.
The image is resized and centre-cropped to {cfg["image_size"]}×{cfg["image_size"]} px, as in training.
Part of [ai-research-labs](https://github.com/ISch-1/ai-research-labs) — Lab C, computer vision.
"""

with gr.Blocks(title="Pet breeds + attribution maps") as demo:
    gr.Markdown("# Pet breed classifier with attribution maps")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Photo", height=320)
            method = gr.Radio(list(METHODS), value="Grad-CAM", label="Attribution method")
            button = gr.Button("Classify", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=5, label="Top 5 breeds (calibrated)")
            heatmap = gr.Image(type="pil", label="Evidence for the top breed", height=320)
            note = gr.Markdown()
    gr.Examples(
        examples=[str(p) for p in sorted((HERE / "examples").glob("*.jpg"), key=_hard_last)],
        inputs=image,
        label="Test images (click one)",
    )
    gr.Markdown(
        "The last example is a known failure — a birman taken for a ragdoll, the most-confused "
        "pair (29 test errors). Test images: Oxford-IIIT Pet dataset (Parkhi et al., 2012), CC BY-SA 4.0."
    )
    button.click(classify, [image, method], [label, heatmap, note])
    image.change(classify, [image, method], [label, heatmap, note])  # upload or example
    method.change(classify, [image, method], [label, heatmap, note])

if __name__ == "__main__":
    demo.launch()
