"""Download the trained models and their configs from the GitHub Release.

Run once, from the project root:
    python scripts/download_models.py

The four .keras files total roughly 500 MB and are not tracked in git — GitHub
rejects files over 100 MB and a repo that size is slow to clone. They live
instead as Release assets and are fetched here into models/<crop>/.
"""

import os
import sys
import urllib.request

BASE_URL = "https://github.com/kaushiksridhar17/crop-doctor/releases/download/v1.0"

FILES = {
    "cassava": ("cassava.keras", "cassava_config.json"),
    "rice":    ("rice.keras",    "rice_config.json"),
    "maize":   ("maize.keras",   "maize_config.json"),
    "tomato":  ("tomato.keras",  "tomato_config.json"),
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url, dest):
    def progress(count, block_size, total_size):
        done = count * block_size
        pct = min(100, 100 * done / total_size) if total_size > 0 else 0
        bar = f"{human_size(done)}/{human_size(total_size)}" if total_size > 0 else human_size(done)
        print(f"\r  {bar}  ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    failures = []

    for crop, (keras_file, config_file) in FILES.items():
        crop_dir = os.path.join(MODELS_DIR, crop)
        os.makedirs(crop_dir, exist_ok=True)

        keras_dest = os.path.join(crop_dir, "best.keras")
        config_dest = os.path.join(crop_dir, "config.json")

        print(f"\n{crop}")
        for filename, dest in ((keras_file, keras_dest), (config_file, config_dest)):
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                print(f"  {filename}: already present, skipping")
                continue
            url = f"{BASE_URL}/{filename}"
            print(f"  downloading {filename} -> {os.path.relpath(dest, ROOT)}")
            try:
                download(url, dest)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                failures.append((crop, filename, str(exc)))

    print()
    if failures:
        print(f"{len(failures)} download(s) failed:")
        for crop, filename, err in failures:
            print(f"  {crop}/{filename}: {err}")
        sys.exit(1)

    print("All four models downloaded and ready.")
    print("Run: python src/app.py")


if __name__ == "__main__":
    main()