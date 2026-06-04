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

## Train Model

```bash
python train.py --data_dir dataset --epochs 10 --save_path model/dpn_mobilenetv2.keras
```

## Predict from Command Line

```bash
python predict.py --model_path model/dpn_mobilenetv2.keras --image_path path/to/image.jpg
```

Output includes:
- Predicted class
- Confidence score
- Raw probability for neuropathy class

## Run Web App

```bash
streamlit run app/streamlit_app.py
```

Then upload a foot image from the browser and view prediction + confidence.

## Notes

- This project is for educational use and not for clinical diagnosis.
- For better accuracy, use a balanced and clean dataset.
