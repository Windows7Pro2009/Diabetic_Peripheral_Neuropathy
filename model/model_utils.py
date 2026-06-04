import tensorflow as tf


IMG_SIZE = (224, 224)


def build_mobilenetv2_model(num_classes=1, train_base=False, learning_rate=1e-4):
    """
    Build a transfer learning model using MobileNetV2 backbone.
    Binary output with sigmoid activation.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = train_base

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model(model, save_path):
    model.save(save_path)


def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)


def decode_foot_prediction(prob_class_index_1):
    """
    Keras ImageDataGenerator uses alphabetical class order:
    neuropathy=0, normal=1. Sigmoid output is P(index 1) = P(normal).
    """
    prob_normal = float(prob_class_index_1)
    prob_neuropathy = 1.0 - prob_normal
    if prob_normal >= 0.5:
        label = "Normal (no neuropathy)"
        confidence = prob_normal
    else:
        label = "Possible diabetic peripheral neuropathy"
        confidence = prob_neuropathy
    return label, confidence, prob_neuropathy
