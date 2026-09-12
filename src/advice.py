"""Treatment guidance for a predicted condition.

The content lives in data/advice.json, deliberately separate from code. It needs
review by a qualified agronomist, and that review should not require reading
Python.

This module's real job is the validation at import: every class in every model
must have an entry, and every entry must match a class. Without that check, a
class added to a model would silently return nothing, and a prediction with no
advice is a prediction a user cannot act on.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVICE_PATH = os.path.join(ROOT, "data", "advice.json")

URGENCY_ORDER = ["none", "monitor", "act this week", "act now", "seek advice"]


def _load() -> dict:
    if not os.path.exists(ADVICE_PATH):
        raise FileNotFoundError(
            f"{ADVICE_PATH} is missing. It holds the guidance shown with every "
            f"prediction.")
    with open(ADVICE_PATH, encoding="utf-8") as f:
        return json.load(f)


ADVICE = _load()


def validate() -> None:
    """Every model class needs advice, and every advice entry needs a class.

    A missing entry means a user gets a diagnosis with nothing to do about it.
    A stray entry means the advice file has drifted from the models.
    """
    problems: List[str] = []

    for crop in registry.CROPS.values():
        entries = ADVICE.get(crop.key)
        if entries is None:
            problems.append(f"{crop.key}: no section in advice.json")
            continue

        for label in crop.class_names:
            entry = entries.get(label)
            if entry is None:
                problems.append(f"{crop.key}/{label}: no advice entry")
                continue

            for required in ("name", "kind", "urgency", "summary", "actions"):
                if required not in entry:
                    problems.append(f"{crop.key}/{label}: missing '{required}'")

            if entry.get("urgency") not in URGENCY_ORDER:
                problems.append(f"{crop.key}/{label}: urgency "
                                f"{entry.get('urgency')!r} is not one of "
                                f"{URGENCY_ORDER}")

            # The kind recorded in the model's own config is authoritative. If
            # the advice file disagrees, one of them is wrong about what the
            # condition actually is — a fungicide for an insect, say.
            model_kind = crop.kinds.get(label)
            if entry.get("kind") != model_kind:
                problems.append(f"{crop.key}/{label}: advice says "
                                f"{entry.get('kind')!r}, model config says "
                                f"{model_kind!r}")

        extra = set(entries) - set(crop.class_names)
        if extra:
            problems.append(f"{crop.key}: advice entries with no matching class: "
                            f"{sorted(extra)}")

    if "_uncertain" not in ADVICE:
        problems.append("no '_uncertain' entry for the abstain case")

    if problems:
        raise ValueError("advice.json does not match the models:\n  " +
                         "\n  ".join(problems))


def get(crop_key: str, label: Optional[str]) -> Dict:
    """Guidance for one prediction. Pass label=None for an abstention."""
    if label is None:
        return dict(ADVICE["_uncertain"])
    return dict(ADVICE[crop_key][label])


def review_status() -> str:
    return ADVICE["_meta"]["review_status"]


validate()


if __name__ == "__main__":
    import sys

    print(f"advice.json validates against all four models\n")
    print(f"REVIEW STATUS: {review_status()}\n")

    if len(sys.argv) >= 3:
        crop_key, label = sys.argv[1], sys.argv[2]
        entry = get(crop_key, None if label == "uncertain" else label)

        print(f"{entry['name']}  ({entry['kind']})")
        print(f"urgency: {entry['urgency']}\n")
        print(entry["summary"] + "\n")
        print("What to do:")
        for a in entry["actions"]:
            print(f"  - {a}")
        if entry.get("prevention"):
            print("\nPrevention:")
            for p in entry["prevention"]:
                print(f"  - {p}")
        if entry.get("note"):
            print(f"\nNote: {entry['note']}")
    else:
        counts = {}
        for crop in registry.CROPS.values():
            for label in crop.class_names:
                u = ADVICE[crop.key][label]["urgency"]
                counts[u] = counts.get(u, 0) + 1
        print("conditions by urgency:")
        for u in URGENCY_ORDER:
            if u in counts:
                print(f"  {u:<18}{counts[u]:>3}")
        print(f"\nusage: python src/advice.py <crop> <label|uncertain>")