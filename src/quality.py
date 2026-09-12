"""Reject unusable photographs before they reach a model.

A classifier handed a blurry, dark, or badly framed image returns a confident
wrong answer rather than an error — it has no way to say "this isn't a leaf".
Checking first prevents a whole class of failure, and telling someone to move
closer costs nothing while a wrong diagnosis costs a spray.

The thresholds are deliberately lenient. Field photographs are taken in a hurry,
in bad light, one-handed, and rejecting a usable image is worse than passing a
marginal one — the abstain threshold is the second line of defence.

**Blur is per-crop, and has to be.** The three source collections differ by an
order of magnitude in measured sharpness:

    rice     median 1062   480x640 originals
    maize    median   78   400x400 crops from larger field photos
    cassava  median   50   Roboflow export, resized and re-encoded

One global threshold cannot serve all three. At 15, cassava lost 29% of images
the model scores 0.954 macro-F1 on; at 5, the gate would do nothing useful for
maize. Each crop's floor sits near its own 3rd percentile and lives in the
registry beside its abstain threshold.

A caveat worth stating plainly: these numbers were calibrated against dataset
images, and a user's phone photo will not resemble any of them. The gate catches
obviously unusable input. It is not a guarantee that a passing image is one the
model was trained to handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from PIL import Image

import registry

ImageInput = Union[str, Image.Image]

DARK_MAX = 45.0          # mean brightness 0-255
BRIGHT_MIN = 225.0       # blown-out
PLANT_MIN = 0.12         # fraction of frame that looks like vegetation
SIZE_MIN = 200           # pixels on the short side
DEFAULT_BLUR_MIN = 15.0  # used when no crop is given


@dataclass
class QualityReport:
    ok: bool
    problems: List[str]
    hints: List[str]
    blur: float
    brightness: float
    plant_fraction: float
    blur_min: float

    @property
    def message(self) -> str:
        if self.ok:
            return ""
        return " ".join(self.hints)


def _laplacian_variance(grey: np.ndarray) -> float:
    """Sharpness proxy. A focused image has strong second derivatives; a blurred
    one does not. Implemented directly to avoid an OpenCV dependency for four
    lines of arithmetic."""
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype="float32")
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        return 0.0

    windows = np.lib.stride_tricks.sliding_window_view(grey, (3, 3))
    response = np.einsum("ijkl,kl->ij", windows, kernel)
    return float(response.var())


def _vegetation_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels that plausibly belong to a plant.

    Green-dominant pixels catch healthy tissue. Diseased tissue is often yellow,
    tan or brown, so a second test picks up warm, saturated pixels that are not
    grey — otherwise a badly diseased leaf would be rejected as "no plant
    visible", which would be exactly backwards.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    total = rgb.shape[0] * rgb.shape[1]

    green = (g > r + 8) & (g > b + 8)

    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    saturation = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    warm = (saturation > 0.25) & (r >= g) & (g > b)

    return float((green | warm).sum() / total)


def check(image: ImageInput, crop_key: Optional[str] = None) -> QualityReport:
    """Assess whether an image is worth classifying.

    Pass crop_key to use that crop's calibrated blur floor. Without it, the
    default applies — fine for a rough check, too strict for cassava.
    """
    blur_min = (registry.get(crop_key).blur_min if crop_key
                else DEFAULT_BLUR_MIN)

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    problems: List[str] = []
    hints: List[str] = []

    width, height = image.size
    if min(width, height) < SIZE_MIN:
        problems.append("too small")
        hints.append(f"This image is {width}x{height}. "
                     f"Use at least {SIZE_MIN} pixels on the short side.")

    # Work at a fixed size so the measurements mean the same thing regardless
    # of what the camera produced.
    work = image.resize((512, 512), Image.BILINEAR)
    rgb = np.asarray(work, dtype="float32")
    grey = rgb.mean(axis=-1)

    blur = _laplacian_variance(grey)
    brightness = float(grey.mean())
    plant = _vegetation_fraction(rgb)

    if blur < blur_min:
        problems.append("blurred")
        hints.append("The photo looks out of focus. Hold steady and tap to focus.")

    if brightness < DARK_MAX:
        problems.append("too dark")
        hints.append("The photo is very dark. Move into better light.")
    elif brightness > BRIGHT_MIN:
        problems.append("overexposed")
        hints.append("The photo is washed out. Avoid shooting into direct sun.")

    if plant < PLANT_MIN:
        problems.append("no plant detected")
        hints.append("Little plant material is visible. "
                     "Move closer so the leaf fills most of the frame.")

    return QualityReport(
        ok=not problems,
        problems=problems,
        hints=hints,
        blur=blur,
        brightness=brightness,
        plant_fraction=plant,
        blur_min=blur_min,
    )


if __name__ == "__main__":
    import glob
    import sys
    from collections import Counter

    if len(sys.argv) < 2:
        print("usage: python src/quality.py <image or glob> [crop] [--quiet]")
        print(f"crops: {', '.join(registry.CROPS)}")
        print("  --quiet  summary only, no per-file lines")
        sys.exit(1)

    quiet = "--quiet" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pattern = args[0]
    crop_key = args[1] if len(args) > 1 else None

    paths = sorted(glob.glob(pattern)) or [pattern]

    if not quiet:
        print(f"{'file':<40}{'blur':>9}{'bright':>9}{'plant':>8}  verdict")
        print("-" * 78)

    reports = []
    problems = Counter()
    unreadable = []

    for path in paths:
        try:
            r = check(path, crop_key)
        except Exception as exc:
            unreadable.append((path, str(exc)[:60]))
            if not quiet:
                name = path.replace("\\", "/").split("/")[-1]
                print(f"{name[:38]:<40}  unreadable")
            continue

        reports.append(r)
        for p in r.problems:
            problems[p] += 1

        if not quiet:
            verdict = "ok" if r.ok else ", ".join(r.problems)
            name = path.replace("\\", "/").split("/")[-1]
            print(f"{name[:38]:<40}{r.blur:>9.0f}{r.brightness:>9.0f}"
                  f"{r.plant_fraction:>8.2f}  {verdict}")

    # ---- summary ----
    total = len(paths)
    passed = sum(1 for r in reports if r.ok)
    failed = len(reports) - passed
    blur_min = reports[0].blur_min if reports else DEFAULT_BLUR_MIN

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {total} files" + (f", crop: {crop_key}" if crop_key
                                        else ", no crop given (default blur floor)"))
    print(f"{'=' * 60}")
    print(f"  passed        {passed:>6}  ({100 * passed / max(total, 1):.1f}%)")
    print(f"  flagged       {failed:>6}  ({100 * failed / max(total, 1):.1f}%)")
    print(f"  unreadable    {len(unreadable):>6}")

    if problems:
        print("\n  reasons flagged (an image can trip more than one):")
        for reason, n in problems.most_common():
            print(f"    {reason:<24}{n:>6}")

    if reports:
        blurs = np.array([r.blur for r in reports])
        brights = np.array([r.brightness for r in reports])
        plants = np.array([r.plant_fraction for r in reports])

        print(f"\n  {'measure':<14}{'min':>9}{'5th pct':>10}{'median':>10}"
              f"{'95th pct':>11}{'max':>10}")
        print("  " + "-" * 62)
        for name, arr, fmt in (
                ("blur", blurs, "{:>9.0f}{:>10.0f}{:>10.0f}{:>11.0f}{:>10.0f}"),
                ("brightness", brights, "{:>9.0f}{:>10.0f}{:>10.0f}{:>11.0f}{:>10.0f}"),
                ("plant frac", plants, "{:>9.2f}{:>10.2f}{:>10.2f}{:>11.2f}{:>10.2f}")):
            print(f"  {name:<14}" + fmt.format(
                arr.min(), np.percentile(arr, 5), np.median(arr),
                np.percentile(arr, 95), arr.max()))

        print(f"\n  thresholds: blur >= {blur_min:.0f}, "
              f"{DARK_MAX:.0f} <= brightness <= {BRIGHT_MIN:.0f}, "
              f"plant >= {PLANT_MIN:.2f}")

        # A gate that rejects a large share of known-good images is
        # miscalibrated, not strict.
        if failed / max(len(reports), 1) > 0.10:
            suggested = np.percentile(blurs, 3)
            print(f"\n  NOTE: {100 * failed / len(reports):.0f}% flagged. If these "
                  f"come from a dataset the model\n  handles well, the threshold "
                  f"is too tight. The 3rd percentile of this\n  set is "
                  f"{suggested:.0f} — consider that as blur_min"
                  + (f" for {crop_key}." if crop_key else "."))

    if unreadable:
        print(f"\n  unreadable files:")
        for path, why in unreadable[:8]:
            print(f"    {path.replace(chr(92), '/').split('/')[-1][:40]:<42}{why}")
        if len(unreadable) > 8:
            print(f"    ... and {len(unreadable) - 8} more")