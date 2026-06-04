# Download proper DFU dataset from Kaggle (recommended)

The best public dataset for this project is:

**Diabetic Foot Ulcer (DFU)** — https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu

~2,673 images with **normal** and **abnormal (ulcer)** classes.

## Setup Kaggle API (one time)

1. Create a Kaggle account and go to **Account → Create New API Token** (downloads `kaggle.json`).
2. Place it at: `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
3. Install CLI: `pip install kaggle`

## Download

```powershell
cd "C:\Users\Mihir\Documents\VSC\Diabetic neuropathy"
kaggle datasets download -d laithjj/diabetic-foot-ulcer-dfu -p data_raw --unzip
```

Then copy images into this project's folders:

- Kaggle **normal** → `dataset/train|val|test/normal/`
- Kaggle **abnormal** or **ulcer** → `dataset/train|val|test/neuropathy/`

Use an 80/10/10 split (image-level). Retrain:

```powershell
python train.py --data_dir dataset --epochs 15
```

## Citation

Cite the DFU dataset authors in your project report (see Kaggle dataset page).
