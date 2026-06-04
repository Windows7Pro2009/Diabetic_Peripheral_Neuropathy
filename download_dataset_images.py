"""
Download additional foot images from Wikimedia Commons (free licenses)
and organize them into dataset/train|val|test/normal|neuropathy.

Usage:
  python download_dataset_images.py --per_class 40
"""

import argparse
import hashlib
import os
import random
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "DPN-College-Project/1.0 (educational; local dataset builder)"

NORMAL_QUERIES = [
    "sole of foot",
    "barefoot sole",
    "human foot sole",
    "plantar foot",
    "healthy foot",
]

NEUROPATHY_QUERIES = [
    "diabetic foot ulcer",
    "diabetic foot",
    "diabetic foot infection",
    "foot ulcer diabetes",
    "neuropathic foot ulcer",
]


def api_get(params):
    params = dict(params)
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        import json

        return json.loads(resp.read().decode("utf-8"))


def search_image_titles(query, limit=50):
    data = api_get(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f'filetype:bitmap {query}',
            "gsrlimit": str(limit),
            "gsrnamespace": "6",
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "iiurlwidth": "512",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    results = []
    for page in pages.values():
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        mime = info.get("mime", "")
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            continue
        url = info.get("thumburl") or info.get("url")
        if url:
            results.append((page.get("title", "file"), url))
    return results


def download_image(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    if len(data) < 5000:
        return False
    with open(dest_path, "wb") as f:
        f.write(data)
    try:
        img = Image.open(dest_path).convert("RGB")
        img = img.resize((224, 224))
        ext = os.path.splitext(dest_path)[1].lower()
        if ext == ".png":
            img.save(dest_path, format="PNG")
        else:
            img.save(dest_path, format="JPEG", quality=90)
        return True
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def unique_name(title, url):
    h = hashlib.md5((title + url).encode()).hexdigest()[:10]
    safe = "".join(c if c.isalnum() else "_" for c in title)[:40]
    return f"wiki_{safe}_{h}.jpg"


def collect_urls(queries, target_count):
    seen = set()
    collected = []
    for q in queries:
        if len(collected) >= target_count:
            break
        try:
            hits = search_image_titles(q, limit=80)
        except Exception as e:
            print(f"  Search failed for '{q}': {e}")
            time.sleep(2)
            continue
        for title, url in hits:
            if url in seen:
                continue
            seen.add(url)
            collected.append((title, url))
            if len(collected) >= target_count:
                break
        time.sleep(1)
    return collected


def split_counts(n, train_ratio=0.8, val_ratio=0.1):
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    return train_n, val_n, test_n


def save_to_splits(items, class_name, root_dir, seed=42):
    rng = random.Random(seed)
    rng.shuffle(items)
    train_n, val_n, test_n = split_counts(len(items))
    splits = [
        ("train", items[:train_n]),
        ("val", items[train_n : train_n + val_n]),
        ("test", items[train_n + val_n :]),
    ]
    saved = 0
    for split, batch in splits:
        out_dir = os.path.join(root_dir, split, class_name)
        os.makedirs(out_dir, exist_ok=True)
        for title, url in batch:
            fname = unique_name(title, url)
            dest = os.path.join(out_dir, fname)
            if os.path.exists(dest):
                saved += 1
                continue
            try:
                if download_image(url, dest):
                    saved += 1
                    print(f"  OK [{split}/{class_name}] {fname}")
                else:
                    print(f"  skip (invalid) {fname}")
            except Exception as e:
                print(f"  fail {fname}: {e}")
            time.sleep(2.5)
    return saved


def count_images(root_dir):
    counts = {}
    for split in ("train", "val", "test"):
        for cls in ("normal", "neuropathy"):
            d = os.path.join(root_dir, split, cls)
            if os.path.isdir(d):
                n = len(
                    [
                        f
                        for f in os.listdir(d)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                    ]
                )
                counts[f"{split}/{cls}"] = n
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikimedia Commons foot images into dataset folders."
    )
    parser.add_argument(
        "--root_dir", type=str, default="dataset", help="Dataset root directory."
    )
    parser.add_argument(
        "--per_class",
        type=int,
        default=40,
        help="Target new images to download per class (approx).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Collecting image URLs from Wikimedia Commons...")
    print("Normal class queries:", NORMAL_QUERIES)
    normal_urls = collect_urls(NORMAL_QUERIES, args.per_class * 2)
    print(f"Found {len(normal_urls)} candidate normal URLs")

    print("Neuropathy class queries:", NEUROPATHY_QUERIES)
    neuro_urls = collect_urls(NEUROPATHY_QUERIES, args.per_class * 2)
    print(f"Found {len(neuro_urls)} candidate neuropathy URLs")

    print("\nDownloading NORMAL images...")
    n_normal = save_to_splits(normal_urls[: args.per_class], "normal", args.root_dir, args.seed)

    print("\nDownloading NEUROPATHY images...")
    n_neuro = save_to_splits(
        neuro_urls[: args.per_class], "neuropathy", args.root_dir, args.seed + 1
    )

    print("\nDone.")
    print(f"Downloaded/kept this run: normal={n_normal}, neuropathy={n_neuro}")
    print("Current dataset counts:")
    for k, v in sorted(count_images(args.root_dir).items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
