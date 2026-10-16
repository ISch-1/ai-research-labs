"""Render the demo's output on its example images as one figure (for the repository README).

    python demos/pets-gradcam/preview.py      # -> docs/assets/pets-demo.png

Not part of the Space (it needs matplotlib, which the demo does not).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inference import HERE, PetClassifier, overlay
from PIL import Image

OUT = HERE.parents[1] / "docs" / "assets" / "pets-demo.png"


def main() -> None:
    clf = PetClassifier()
    paths = sorted(
        (HERE / "examples").glob("*.jpg"), key=lambda p: (p.stem.endswith("_hard"), p.stem)
    )
    paths = [p for p in paths if p.stem in {"bengal", "keeshond", "beagle"}] + [
        p for p in paths if p.stem.endswith("_hard")
    ]
    fig, axes = plt.subplots(3, len(paths), figsize=(2.6 * len(paths), 8.4))
    for col, path in enumerate(paths):
        crop, x = clf.preprocess(Image.open(path))
        probs = clf.probabilities(x)
        top = probs.topk(2)
        k = int(top.indices[0])
        truth = path.stem.removesuffix("_hard").replace("_", " ").title()
        title = f"true: {truth}\n" + "\n".join(
            f"{clf.classes[int(i)]} {float(p):.0%}"
            for p, i in zip(top.values, top.indices, strict=True)
        )
        panels = [
            (crop, title),
            (overlay(crop, clf.grad_cam(x, k)), "Grad-CAM"),
            (overlay(crop, clf.integrated_gradients(x, k), smooth=5), "Integrated Gradients"),
        ]
        for row, (image, label) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_title(
                label,
                fontsize=9,
                color="firebrick" if row == 0 and path.stem.endswith("_hard") else "black",
            )
            ax.axis("off")
    fig.suptitle(
        "Pet breed demo — calibrated top-2 and the evidence for the top breed", fontsize=11
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(OUT)


if __name__ == "__main__":
    main()
