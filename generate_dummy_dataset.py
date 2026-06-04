import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_normal_image(size=224):
    """
    Create a synthetic 'normal' foot-like image:
    smoother tones, fewer dark lesions.
    """
    base = np.ones((size, size, 3), dtype=np.uint8) * np.array([220, 185, 160], dtype=np.uint8)
    noise = np.random.normal(0, 7, (size, size, 3)).astype(np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    # Draw faint vessel-like lines
    for _ in range(random.randint(4, 8)):
        x1, y1 = random.randint(0, size - 1), random.randint(0, size - 1)
        x2, y2 = random.randint(0, size - 1), random.randint(0, size - 1)
        color = (200, 150, 130)
        draw.line((x1, y1, x2, y2), fill=color, width=1)

    return pil


def make_neuropathy_image(size=224):
    """
    Create a synthetic 'neuropathy' image:
    includes random dark circular patches and stronger texture.
    """
    base = np.ones((size, size, 3), dtype=np.uint8) * np.array([205, 165, 145], dtype=np.uint8)
    noise = np.random.normal(0, 12, (size, size, 3)).astype(np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    # Add random dark lesion-like spots
    for _ in range(random.randint(5, 12)):
        r = random.randint(6, 20)
        x = random.randint(r, size - r)
        y = random.randint(r, size - r)
        shade = random.randint(40, 100)
        color = (shade, shade - 10 if shade > 10 else shade, shade - 20 if shade > 20 else shade)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    # Add few reddish irritation-like spots
    for _ in range(random.randint(2, 6)):
        r = random.randint(8, 18)
        x = random.randint(r, size - r)
        y = random.randint(r, size - r)
        color = (165, 70, 70)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    return pil


def save_images(target_dir, class_name, count, size):
    ensure_dir(target_dir)
    for i in range(count):
        if class_name == "normal":
            img = make_normal_image(size=size)
        else:
            img = make_neuropathy_image(size=size)
        img.save(os.path.join(target_dir, f"{class_name}_{i + 1:03d}.png"))


def generate_dataset(root_dir, train_count, val_count, test_count, size):
    splits = {
        "train": train_count,
        "val": val_count,
        "test": test_count,
    }
    classes = ["normal", "neuropathy"]

    for split, count in splits.items():
        for cls in classes:
            split_class_dir = os.path.join(root_dir, split, cls)
            save_images(split_class_dir, cls, count, size)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dummy dataset for DPN project.")
    parser.add_argument("--root_dir", type=str, default="dataset", help="Dataset root directory.")
    parser.add_argument("--train_count", type=int, default=40, help="Images per class in train split.")
    parser.add_argument("--val_count", type=int, default=10, help="Images per class in val split.")
    parser.add_argument("--test_count", type=int, default=10, help="Images per class in test split.")
    parser.add_argument("--size", type=int, default=224, help="Image size (size x size).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    generate_dataset(
        root_dir=args.root_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        size=args.size,
    )

    print("Dummy dataset created successfully.")
    print(f"Location: {args.root_dir}")
    print("Structure: train/val/test with normal and neuropathy classes.")


if __name__ == "__main__":
    main()
