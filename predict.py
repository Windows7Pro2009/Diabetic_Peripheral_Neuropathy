import argparse

import numpy as np

from model.model_utils import decode_foot_prediction, load_trained_model
from utils.preprocess import load_and_preprocess_image


def predict_image(model_path, image_path):
    model = load_trained_model(model_path)
    image_tensor = load_and_preprocess_image(image_path)
    prob = float(model.predict(image_tensor, verbose=0)[0][0])
    return decode_foot_prediction(prob)


def main():
    parser = argparse.ArgumentParser(description="Predict DPN from foot image.")
    parser.add_argument("--model_path", type=str, default="model/dpn_mobilenetv2.keras")
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()

    label, confidence, raw_prob = predict_image(args.model_path, args.image_path)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Raw neuropathy probability: {raw_prob:.4f}")


if __name__ == "__main__":
    main()
