import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import Xception

# Define the Encoder Model Architecture
def get_encoder(input_shape=(128, 128, 3)):
    pretrained_model = Xception(input_shape=input_shape, weights='imagenet', include_top=False, pooling='avg')
    for layer in pretrained_model.layers[:-27]:  # Freeze initial layers for transfer learning
        layer.trainable = False
    return Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")

# Load Encoder Model Weights
def load_encoder_model(weights_path='./models/encoder.h5'):
    encoder = get_encoder()
    encoder.load_weights(weights_path)
    print("Encoder weights loaded successfully.")
    return encoder

encoder = load_encoder_model()
