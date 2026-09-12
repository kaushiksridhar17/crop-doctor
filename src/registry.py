"""Crop definitions — the single place that knows about the four models.

Every model was trained separately, so they differ in architecture, input size,
class list, and calibrated threshold. Nothing outside this module should need to
know that.

Two per-crop numbers here were set by measurement rather than chosen:

  threshold — from a sweep on that crop's validation split, never test. The
    figures beside it are what it costs and buys.

  blur_min — from the distribution of measured sharpness in that crop's source
    images, at roughly the 3rd percentile. The three collections differ by an
    order of magnitude, so a single global floor rejects a quarter of one
    dataset while doing nothing for another.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


# The rice model was trained before display names and kinds were written into
# config.json, so they live here instead. Everything else reads its own config,
# which cannot drift from the model it describes.
RICE_DISPLAY = {
    "bacterial_leaf_blight": "bacterial leaf blight",
    "bacterial_leaf_streak": "bacterial leaf streak",
    "bacterial_panicle_blight": "bacterial panicle blight",
    "blast": "blast",
    "brown_spot": "brown spot",
    "dead_heart": "dead heart (stem borer)",
    "downy_mildew": "downy mildew",
    "hispa": "hispa (beetle)",
    "normal": "healthy",
    "tungro": "tungro",
}

RICE_KINDS = {
    "bacterial_leaf_blight": "bacterial disease",
    "bacterial_leaf_streak": "bacterial disease",
    "bacterial_panicle_blight": "bacterial disease",
    "blast": "fungal disease",
    "brown_spot": "fungal disease",
    "dead_heart": "insect pest",
    "downy_mildew": "fungal disease",
    "hispa": "insect pest",
    "normal": "no problem detected",
    "tungro": "viral disease",
}

FALLBACK_METADATA = {
    "rice": (RICE_DISPLAY, RICE_KINDS),
}


@dataclass
class Crop:
    key: str
    name: str
    img_size: int
    threshold: float          # below this, abstain
    refer_pct: float          # what the threshold refers, from validation
    accuracy_kept: float      # accuracy on what it still answers
    test_macro_f1: float      # headline number on the held-out test split
    background_drop: float    # accuracy points lost when background is replaced
    blur_min: float           # sharpness floor; see quality.py
    source: str

    class_names: Optional[List[str]] = field(default=None)
    display: Optional[Dict[str, str]] = field(default=None)
    kinds: Optional[Dict[str, str]] = field(default=None)

    @property
    def model_path(self) -> str:
        return os.path.join(MODELS_DIR, self.key, "best.keras")

    @property
    def config_path(self) -> str:
        return os.path.join(MODELS_DIR, self.key, "config.json")


CROPS: Dict[str, Crop] = {
    "cassava": Crop(
        key="cassava", name="Cassava", img_size=320,
        threshold=0.80, refer_pct=7.1, accuracy_kept=0.9824,
        test_macro_f1=0.9537, background_drop=6.1,
        # Roboflow export: resized and re-encoded, which smooths edges.
        # Median measured blur 50, 3rd percentile ~5.
        blur_min=5.0,
        source="Roboflow cassava dataset, 13,187 images, provided split",
    ),
    "rice": Crop(
        key="rice", name="Rice (paddy)", img_size=320,
        threshold=0.70, refer_pct=27.0, accuracy_kept=0.9763,
        test_macro_f1=0.8884,
        # Not measured. The background diagnostic was added after this model
        # was trained. Shown as "not measured" rather than guessed.
        background_drop=float("nan"),
        # 480x640 originals, sharp. Median measured blur 1062.
        blur_min=100.0,
        source="Paddy Doctor, 10,407 field images, Tamil Nadu",
    ),
    "maize": Crop(
        key="maize", name="Maize", img_size=320,
        threshold=0.80, refer_pct=23.0, accuracy_kept=0.9707,
        test_macro_f1=0.8956, background_drop=4.6,
        # 400x400 crops from larger field photos. Median measured blur 78.
        blur_min=15.0,
        source="CCMT raw, field images, Ghana",
    ),
    "tomato": Crop(
        key="tomato", name="Tomato", img_size=400,
        threshold=0.70, refer_pct=28.0, accuracy_kept=0.9010,
        test_macro_f1=0.8021, background_drop=14.5,
        blur_min=15.0,          # same collection and processing as maize
        source="CCMT raw, field images, Ghana",
    ),
}


def load_metadata() -> None:
    """Read class names and display strings from each model's config.json."""
    for crop in CROPS.values():
        if not os.path.exists(crop.config_path):
            raise FileNotFoundError(
                f"{crop.config_path} is missing. Each crop needs best.keras and "
                f"config.json under models/<crop>/.")

        with open(crop.config_path) as f:
            cfg = json.load(f)

        crop.class_names = cfg["class_names"]

        fallback_display, fallback_kinds = FALLBACK_METADATA.get(crop.key, ({}, {}))
        crop.display = cfg.get("display_names") or fallback_display or \
            {c: c for c in crop.class_names}
        crop.kinds = cfg.get("kinds") or fallback_kinds or \
            {c: "unknown" for c in crop.class_names}

        missing = set(crop.class_names) - set(crop.kinds)
        if missing:
            raise ValueError(f"{crop.key}: no kind recorded for {sorted(missing)}")

        # The config records what the model was trained at. If that disagrees
        # with the registry, the registry is wrong and predictions would be
        # silently resized incorrectly.
        recorded = cfg.get("img_size")
        if recorded and recorded != crop.img_size:
            raise ValueError(
                f"{crop.key}: registry says img_size={crop.img_size}, but "
                f"config.json says {recorded}. Fix the registry.")


def get(key: str) -> Crop:
    if key not in CROPS:
        raise KeyError(f"Unknown crop {key!r}. Choose from {sorted(CROPS)}.")
    return CROPS[key]


load_metadata()


if __name__ == "__main__":
    for crop in CROPS.values():
        drop = ("not measured" if crop.background_drop != crop.background_drop
                else f"{crop.background_drop:.1f} pts")
        print(f"\n{crop.name}  ({crop.key})")
        print(f"  {crop.img_size}px, {len(crop.class_names)} classes")
        print(f"  abstain below {crop.threshold:.2f} "
              f"(refers {crop.refer_pct:.0f}%, "
              f"accuracy on the rest {crop.accuracy_kept:.4f})")
        print(f"  test macro-F1 {crop.test_macro_f1:.4f}, background drop {drop}")
        print(f"  blur floor {crop.blur_min:.0f}")
        print(f"  {crop.source}")
        for c in crop.class_names:
            print(f"    {crop.display[c]:<26} {crop.kinds[c]}")