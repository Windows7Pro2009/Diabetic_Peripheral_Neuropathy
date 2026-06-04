"""
Augment existing dataset images to increase training data.
Creates flipped/rotated/color-varied copies alongside originals.

Usage:
  python augment_dataset.py --copies_per_image 25
"""

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def augment_one(img):
    """Apply random augmentation; returns RGB PIL Image."""
    img = img.convert("RGB")
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageOps.flip(img)
    angle = random.choice([0, 90, 180, 270, -15, 15, -25, 25])
    if angle:
        img = img.rotate(angle, fillcolor=(128, 128, 128))
    scale = random.uniform(0.85, 1.0)
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)))
    img = img.resize((224, 224))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))
    return img


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts) and not f.startswith("aug_")
    ]


def augment_folder(folder, copies_per_image, seed):
    rng = random.Random(seed)
    paths = list_images(folder)
    created = 0
    for src in paths:
        base = os.path.splitext(os.path.basename(src))[0]
        ext = os.path.splitext(src)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        try:
            original = Image.open(src)
        except Exception:
            continue
        for i in range(copies_per_image):
            rng.seed(seed + hash(src) % 10000 + i)
            out_name = f"aug_{base}_{i:03d}{ext}"
            out_path = os.path.join(folder, out_name)
            if os.path.exists(out_path):
                created += 1
                continue
            try:
                aug = augment_one(original)
                aug.save(out_path, quality=90)
                created += 1
            except Exception:
                pass
    return created


def count_all(root):
    total = {}
    for split in ("train", "val", "test"):
        for cls in ("normal", "neuropathy"):
            d = os.path.join(root, split, cls)
            if os.path.isdir(d):
                n = len(list_images(d)) + len(
                    [f for f in os.listdir(d) if f.startswith("aug_")]
                )
                total[f"{split}/{cls}"] = n
    return total


def main():
    parser = argparse.ArgumentParser(description="Augment dataset images in-place.")
    parser.add_argument("--root_dir", type=str, default="dataset")
    parser.add_argument(
        "--copies_per_image",
        type=int,
        default=25,
        help="Augmented copies per source image.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Before:")
    for k, v in sorted(count_all(args.root_dir).items()):
        print(f"  {k}: {v}")

    total_created = 0
    for split in ("train", "val", "test"):
        for cls in ("normal", "neuropathy"):
            folder = os.path.join(args.root_dir, split, cls)
            if not os.path.isdir(folder):
                continue
            n = augment_folder(folder, args.copies_per_image, args.seed)
            total_created += n
            print(f"Augmented {split}/{cls}: {n} files present")

    print(f"\nTotal files in dataset (including aug_*): {sum(count_all(args.root_dir).values())}")
    print("After:")
    for k, v in sorted(count_all(args.root_dir).items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
