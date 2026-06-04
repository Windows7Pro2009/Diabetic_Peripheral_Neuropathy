from PIL import Image
import numpy as np
import tensorflow as tf


IMG_SIZE = (224, 224)


def load_and_preprocess_image(image_path, target_size=IMG_SIZE):
    """
    Load image from disk and apply preprocessing for MobileNetV2.
    Returns a batch tensor with shape (1, 224, 224, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_uploaded_pil_image(pil_image, target_size=IMG_SIZE):
    """
    Preprocess an uploaded PIL image for web inference.
    """
    img = pil_image.convert("RGB").resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
