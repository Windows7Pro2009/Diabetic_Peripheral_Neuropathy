import argparse
import json
import os
import tensorflow as tf

from model.model_utils import build_mobilenetv2_model, save_model


IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def create_data_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
    )
    return train_gen, val_gen


def main():
    parser = argparse.ArgumentParser(description="Train DPN classifier model.")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset root path.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="model/dpn_mobilenetv2.keras",
        help="Path to save the trained model.",
    )
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of head warmup epochs.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs.")
    args = parser.parse_args()

    train_gen, val_gen = create_data_generators(args.data_dir)

    print("\n--- STAGE 1: Training Classification Head (Feature Extraction) ---")
    # Build model with pre-trained base frozen (train_base=False) and warm-up learning rate (1e-4)
    model = build_mobilenetv2_model(train_base=False, learning_rate=1e-4)
    model.summary()

    callbacks_warmup = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.warmup_epochs,
        callbacks=callbacks_warmup,
    )

    print("\n--- STAGE 2: Fine-Tuning Top Layers of MobileNetV2 ---")
    # Unfreeze base model from layer 100 onwards
    base_model = None
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2") or isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is not None:
        print(f"Unfreezing layers in {base_model.name} from layer 100 onwards...")
        base_model.trainable = True
        # Keep early layers frozen (lower-level feature extractors like edge detectors)
        for layer in base_model.layers[:100]:
            layer.trainable = False
        for layer in base_model.layers[100:]:
            layer.trainable = True
    else:
        print("Warning: MobileNetV2 base model layer not found. Fine-tuning all layers.")
        # Fallback to unfreezing the whole model if functional layer not found
        model.trainable = True

    # Re-compile with a much smaller learning rate to prevent destroying pre-trained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks_finetune = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.save_path, monitor="val_loss", save_best_only=True
        ),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks_finetune,
    )

    # Save the final model (in case check-pointed one isn't saved, or to keep final weights)
    save_model(model, args.save_path)
    
    # Save the class indices mapping for inference decoding
    indices_path = os.path.join(os.path.dirname(args.save_path), "class_indices.json")
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2)
        
    print(f"\nModel saved at: {args.save_path}")
    print(f"Class indices: {train_gen.class_indices}")
    print(f"Class indices saved at: {indices_path}")
    print("Note: sigmoid output = P(class index 1). Use decode_foot_prediction() at inference.")


if __name__ == "__main__":
    main()
