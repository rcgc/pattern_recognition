import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import argparse
from utils import build_model  # Import model from utils.py

# Function to load image data
def load_data(batch_size=32):
    """Loads training and test data generators from the 'training' folder."""
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        "training", target_size=(64, 64), color_mode="grayscale",
        batch_size=batch_size, class_mode='sparse', subset='training')

    validation_generator = datagen.flow_from_directory(
        "training", target_size=(64, 64), color_mode="grayscale",
        batch_size=batch_size, class_mode='sparse', subset='validation')

    return train_generator, validation_generator

# Function to plot training history
def plot_training_history(history):
    """Plots training and validation accuracy/loss."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    plt.show()

# Function to compute and plot confusion matrix
def plot_confusion_matrix(model, generator, title):
    """Computes and plots confusion matrix."""
    y_true = generator.classes
    y_pred = np.argmax(model.predict(generator), axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

# Main function
def main(epochs, batch_size):
    train_generator, validation_generator = load_data(batch_size)

    model = build_model()
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    plot_training_history(history)
    plot_confusion_matrix(model, train_generator, "Training Confusion Matrix")

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/mnist_cnn_model.h5")
    print("Model saved successfully in 'models/mnist_cnn_model.h5'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    main(args.epochs, args.batch_size)
