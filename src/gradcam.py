"""Grad-CAM — which regions of the image drove the prediction.

Not decoration. This is how two broken models were caught during development:
one scoring 0.979 on PlantVillage tomato was attending to the grey card behind
the leaf, and another scoring 0.986 on a curated maize collection was reading
the black corners left by rotation. Both would have looked fine from their
accuracy alone.

For a user, it answers the question accuracy cannot: is this answer worth
trusting on *this* photograph?
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import keras
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image

import registry
from predict import ImageInput, get_model, to_batch

# Splitting a model is slow, so each crop's split is computed once and cached.
_splits: Dict[str, tuple] = {}


def split_model(model: keras.Model) -> Tuple[list, keras.Model, list]:
    """Split into preprocessing, backbone-up-to-last-conv, and head.

    The backbone is a nested Functional model, so its internal layer outputs
    belong to its own graph rather than the outer one. Building a keras.Model
    from the outer input to an inner layer therefore fails. Running the three
    pieces by hand sidesteps that.
    """
    backbone = next((l for l in model.layers
                     if isinstance(l, keras.Model) and len(l.layers) > 10), None)
    if backbone is None:
        raise ValueError("No nested backbone found in this model")

    conv_name = None
    for layer in reversed(backbone.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if len(shape) == 4:
            conv_name = layer.name
            break
    if conv_name is None:
        raise ValueError("No 4D convolutional layer found in the backbone")

    index = model.layers.index(backbone)
    return (model.layers[1:index],
            keras.Model(backbone.input, backbone.get_layer(conv_name).output),
            model.layers[index + 1:])


def get_split(crop_key: str) -> tuple:
    if crop_key not in _splits:
        _splits[crop_key] = split_model(get_model(crop_key))
    return _splits[crop_key]


def compute_heatmap(batch: np.ndarray, crop_key: str,
                    class_index: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Return a 2D heatmap in 0-1 and the class index it explains.

    This forward pass through the backbone is unavoidable for Grad-CAM — it is
    how the conv features and the gradient are obtained. What is avoidable is a
    *separate* plain classification pass just to pick class_index; callers that
    already know the predicted class (predict.py does) should pass it in.
    """
    pre_layers, conv_submodel, post_layers = get_split(crop_key)

    x = batch
    for layer in pre_layers:
        x = layer(x)

    with tf.GradientTape() as tape:
        conv_out = conv_submodel(x, training=False)
        tape.watch(conv_out)          # a tensor, not a variable — must watch it
        y = conv_out
        for layer in post_layers:
            y = layer(y, training=False)
        if class_index is None:
            class_index = int(tf.argmax(y[0]))
        score = y[:, class_index]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)                 # positive contributions only
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), class_index


def overlay(image: ImageInput, heatmap: np.ndarray,
            img_size: int, alpha: float = 0.5) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    base = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)

    coloured = cm.jet(heatmap)[..., :3]
    coloured = Image.fromarray(np.uint8(coloured * 255)).resize(
        base.size, Image.BILINEAR)
    return Image.blend(base, coloured, alpha)


def explain(image: ImageInput, crop_key: str,
            class_index: Optional[int] = None,
            batch: Optional[np.ndarray] = None) -> Image.Image:
    """Produce the attention overlay for one image.

    Pass batch and class_index if the caller already has them — predict.py
    does — so this skips re-preprocessing the image and lets compute_heatmap
    explain the class that was actually reported, rather than running its own
    independent argmax that could in principle differ.
    """
    crop = registry.get(crop_key)
    if batch is None:
        batch = to_batch(image, crop.img_size)
    heatmap, _ = compute_heatmap(batch, crop_key, class_index)
    return overlay(image, heatmap, crop.img_size)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python src/gradcam.py <crop> <image> [output.jpg]")
        print(f"crops: {', '.join(registry.CROPS)}")
        sys.exit(1)

    crop_key, path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "gradcam_output.jpg"

    from predict import predict

    result = predict(path, crop_key)
    explain(path, crop_key).save(out_path)

    print()
    if result["abstained"]:
        print(f"UNCERTAIN — closest match {result['top_display']} at "
              f"{result['confidence']:.1%}")
    else:
        print(f"{result['display_name']}  ({result['confidence']:.1%})")
    print(f"heatmap saved to {out_path}")