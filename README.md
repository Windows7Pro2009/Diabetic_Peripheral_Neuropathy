# AI-Based Detection of Diabetic Peripheral Neuropathy

This is a simple college mini project that classifies foot images into:
- **Normal (no neuropathy)**
- **Possible diabetic peripheral neuropathy**

It uses **TensorFlow + MobileNetV2 transfer learning** and includes:
- Training script
- Model save/load
- Prediction script with confidence score
- Streamlit web app for image upload and result display

## Project Structure

```
Diabetic neuropathy/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
├── dataset/
│   └── README.md
├── model/
│   ├── README.md
│   └── model_utils.py
├── utils/
│   ├── __init__.py
│   └── preprocess.py
├── predict.py
├── requirements.txt
└── train.py
```

## Dataset Format

Arrange your dataset like:

```
dataset/
  train/
    normal/
    neuropathy/
  val/
    normal/
    neuropathy/
  test/
    normal/
    neuropathy/
```

## Setup

1. Create and activate virtual environment (recommended):
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> You should see `(.venv)` appear at the start of your terminal prompt.

---

3. (One-time) Download & Set Up the Dataset

> Skip this step if the `dataset/` folder already has images inside it.

```powershell
python setup_dataset.py
```

This will:
1. Download the **Diabetic Foot Ulcer (DFU)** dataset from Kaggle using your saved API token
2. Extract and organize the images automatically
3. Split them into `train` / `val` / `test` folders (80% / 10% / 10%)

**Expected output structure:**
```
dataset/
  train/
    normal/       (~434 images)
    neuropathy/   (~409 images)
  val/
    normal/       (~54 images)
    neuropathy/   (~51 images)
  test/
    normal/       (~55 images)
    neuropathy/   (~52 images)
```

---

4. (One-time) Train the Model

> Skip this step if `model/dpn_mobilenetv2.keras` already exists.

```powershell
python train.py --warmup_epochs 5 --epochs 15
```

This runs a **two-stage transfer learning** process:

| Stage | What Happens | Epochs |
|---|---|---|
| Stage 1 (Warm-up) | Trains only the classification head on frozen MobileNetV2 | 5 |
| Stage 2 (Fine-tuning) | Unlocks top MobileNetV2 layers, trains with a tiny learning rate | Up to 15 |

The best model is saved to `model/dpn_mobilenetv2.keras`.

> **Training time:** ~12–15 minutes on CPU.

---

5. Run the Web App

```powershell
streamlit run app/streamlit_app.py
```

Then open your browser and go to: **http://localhost:8501**

Upload any foot image (JPG/PNG) to see:
- ✅ Predicted class: **Normal** or **Possible diabetic peripheral neuropathy**
- 📊 Confidence score (%)
- 🔢 Raw neuropathy probability

---

6.  Predict from the Command Line (Optional)

```powershell
python predict.py --image_path path\to\your\image.jpg
```

**Example:**
```powershell
python predict.py --image_path dataset\test\normal\1.jpg
```

**Sample results from the trained model:**

| Image | Prediction | Confidence |
|---|---|---|
| `test/normal/1.jpg` | Normal (no neuropathy) | **96.95%** |
| `test/neuropathy/11.jpg` | Possible diabetic peripheral neuropathy | **99.52%** |

---

## Quick Reference

```powershell
# ── First-time setup ──────────────────────────────
.\.venv\Scripts\Activate.ps1
python setup_dataset.py
python train.py --warmup_epochs 5 --epochs 15

# ── Everyday use ──────────────────────────────────
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py

# ── Single image prediction ───────────────────────
.\.venv\Scripts\Activate.ps1
python predict.py --image_path path\to\image.jpg
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.9 or higher |
| Virtual environment | `.venv/` in project root |
| Kaggle API token | `~/.kaggle/kaggle.json` (for dataset download) |
| Dependencies | Installed via `pip install -r requirements.txt` |

---

## Notes

- This project is for **educational purposes only** and is **not for clinical diagnosis**.
- The model was trained on the [Diabetic Foot Ulcer (DFU)](https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu) dataset by Alzubaidi et al.
- TensorFlow will show GPU-related warnings on Windows — these are harmless and can be ignored.
