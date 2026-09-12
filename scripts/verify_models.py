"""Load all four models and run one prediction each.

Worth doing before anything else. Each model was saved by a different notebook,
and the custom preprocessing layer is registered under a different package name
depending on which notebook wrote it, so this is where a mismatch surfaces.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import keras
from keras import layers

import registry


# Every package name used across the training notebooks is registered here, so
# any of the four saved models can be rebuilt. The name must match whatever the
# notebook used or load_model cannot find the class.
def _register(package):
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

    return Preprocess


for pkg in ("cropdx", "cassava", "paddy"):
    _register(pkg)


def main():
    ok = True
    for crop in registry.CROPS.values():
        print(f"\n{crop.name}")

        if not os.path.exists(crop.model_path):
            print(f"  MISSING: {crop.model_path}")
            ok = False
            continue

        size_mb = os.path.getsize(crop.model_path) / 1024 / 1024
        start = time.perf_counter()
        try:
            model = keras.models.load_model(crop.model_path)
        except Exception as exc:
            print(f"  FAILED TO LOAD: {type(exc).__name__}")
            print(f"  {exc}")
            ok = False
            continue
        load_s = time.perf_counter() - start

        n_out = model.output_shape[-1]
        in_size = model.input_shape[1]

        print(f"  loaded in {load_s:.1f}s, {size_mb:.0f} MB, "
              f"{model.count_params():,} params")
        print(f"  input {model.input_shape}, output {model.output_shape}")

        # The registry and the model must agree, or every prediction is
        # silently resized wrong or indexed into the wrong class list.
        if in_size != crop.img_size:
            print(f"  MISMATCH: registry says {crop.img_size}px, model wants {in_size}px")
            ok = False
        if n_out != len(crop.class_names):
            print(f"  MISMATCH: registry lists {len(crop.class_names)} classes, "
                  f"model outputs {n_out}")
            ok = False

        probe = np.random.uniform(0, 255, (1, in_size, in_size, 3)).astype("float32")
        model.predict(probe, verbose=0)          # warm up, then time it
        start = time.perf_counter()
        probs = model.predict(probe, verbose=0)[0]
        ms = 1000 * (time.perf_counter() - start)

        print(f"  inference {ms:.0f} ms, probabilities sum to {probs.sum():.4f}")
        if abs(probs.sum() - 1.0) > 0.01:
            print("  WARNING: output does not look like a softmax")
            ok = False

        del model
        keras.backend.clear_session()

    print("\n" + ("all four models load and predict" if ok
                  else "PROBLEMS FOUND — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())