import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
def load_data():
    (train_images, train_labels), _ = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255  # Normalize
    train_labels = to_categorical(train_labels)
    return train_images[:1000], train_labels[:1000]  # Use a subset for quick training

# Define the model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Preprocess uploaded image for prediction
def preprocess_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    img_array = img_array.reshape((1, 28, 28, 1)).astype('float32') / 255  # Normalize
    return img_array

# Predict digit from image
def predict_digit(model, img):
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)

