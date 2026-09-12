"""Prediction with lazy model loading and per-crop abstention.

Models are loaded on first use and cached, rather than all four at startup.
Each is roughly 110-140 MB in memory, and a user working on one crop should not
pay for the other three.
"""

from __future__ import annotations

import io
from typing import Dict, Optional, Union

import keras
import numpy as np
from keras import layers
from PIL import Image

import registry


# The custom preprocessing layer is registered under a different package name
# depending on which notebook saved the model. All three are registered so any
# of the four can be rebuilt by load_model.
def _register(package: str):
    @keras.saving.register_keras_serializable(package=package)
    class Preprocess(layers.Layer):
        def __init__(self, module=None, **kwargs):
            super().__init__(**kwargs)
            self.module = module

        def call(self, inputs):
            if self.module is None:
                return inputs
            return getattr(keras.applications, self.module).preprocess_input(inputs)

        def get_config(self):
            c = super().get_config()
            c.update({"module": self.module})
            return c


for _pkg in ("cropdx", "cassava", "paddy"):
    _register(_pkg)


_models: Dict[str, keras.Model] = {}

ImageInput = Union[str, io.BytesIO, Image.Image]


def get_model(crop_key: str) -> keras.Model:
    """Load a crop's model once and reuse it."""
    if crop_key not in _models:
        crop = registry.get(crop_key)
        print(f"[predict] loading {crop.name}...")
        model = keras.models.load_model(crop.model_path)

        # Warm up so the first real request does not pay the graph-tracing cost,
        # which is several seconds.
        model.predict(
            np.zeros((1, crop.img_size, crop.img_size, 3), dtype="float32"),
            verbose=0)
        _models[crop_key] = model
        print(f"[predict] {crop.name} ready")
    return _models[crop_key]


def to_batch(image: ImageInput, img_size: int) -> np.ndarray:
    """Accept a path, a file object, or a PIL image; return a model-ready batch.

    Values stay in 0-255. Every model applies its backbone's own preprocessing
    as its first layer, so rescaling here would apply it twice.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    return np.expand_dims(np.asarray(image, dtype="float32"), 0)


def classify_batch(batch: np.ndarray, crop_key: str) -> np.ndarray:
    """Run the model once and return raw probabilities for a pre-built batch.

    Exists so predict() and gradcam.explain() can share one forward pass instead
    of each doing their own. Grad-CAM needs its own forward pass through the
    backbone regardless — that is how it gets the conv features — but nothing
    upstream of that (preprocessing, the plain classification pass) needs to
    happen twice at every diagnosis.
    """
    model = get_model(crop_key)
    return model.predict(batch, verbose=0)[0]


def predict(image: ImageInput, crop_key: str,
            threshold: Optional[float] = None) -> dict:
    """Classify an image of a known crop.

    Returns 'uncertain' rather than a label when confidence falls below the
    crop's calibrated threshold. A confident wrong answer costs a farmer a
    spray, a season, or a field; an honest referral costs a phone call.
    """
    crop = registry.get(crop_key)
    threshold = crop.threshold if threshold is None else threshold

    batch = to_batch(image, crop.img_size)
    probs = classify_batch(batch, crop_key)

    best = int(np.argmax(probs))
    label = crop.class_names[best]
    confidence = float(probs[best])
    abstained = confidence < threshold

    return {
        "crop": crop.key,
        "crop_name": crop.name,
        "label": None if abstained else label,
        "display_name": ("Uncertain — recommend expert review" if abstained
                         else crop.display[label]),
        "kind": None if abstained else crop.kinds[label],
        "confidence": confidence,
        "abstained": abstained,
        "top_class": label,                  # the suppressed guess, for logging
        "top_display": crop.display[label],
        "threshold": threshold,
        "all_scores": {crop.display[c]: float(p)
                       for c, p in zip(crop.class_names, probs)},
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python src/predict.py <crop> <image>")
        print(f"crops: {', '.join(registry.CROPS)}")
        sys.exit(1)

    crop_key, path = sys.argv[1], sys.argv[2]
    result = predict(path, crop_key)

    print()
    if result["abstained"]:
        print(f"UNCERTAIN — closest match {result['top_display']} at "
              f"{result['confidence']:.1%}, below the "
              f"{result['threshold']:.0%} threshold")
        print("Recommend expert review.")
    else:
        print(f"{result['display_name']}  ({result['confidence']:.1%})")
        print(f"{result['kind']}")

    print()
    for name, score in sorted(result["all_scores"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<28} {score:.3f}")