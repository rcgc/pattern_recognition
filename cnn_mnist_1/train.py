import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import argparse
from utils import build_model  # Importing the model builder function

# Function to load image data
def load_data(batch_size=32):
    """Loads training, validation, and test data generators."""
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "training", target_size=(64, 64), color_mode="grayscale",
        batch_size=batch_size, class_mode='sparse', subset='training')

    validation_generator = train_datagen.flow_from_directory(
        "training", target_size=(64, 64), color_mode="grayscale",
        batch_size=batch_size, class_mode='sparse', subset='validation')

    test_generator = test_datagen.flow_from_directory(
        "testing", target_size=(64, 64), color_mode="grayscale",
        batch_size=batch_size, class_mode='sparse', shuffle=False)

    return train_generator, validation_generator, test_generator

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

# Function to compute and plot confusion matrices
def plot_confusion_matrices(model, train_generator, test_generator):
    """Computes and plots confusion matrices for training and test sets."""
    y_true_train = train_generator.classes
    y_pred_train = np.argmax(model.predict(train_generator), axis=1)
    train_cm = confusion_matrix(y_true_train, y_pred_train)

    y_true_test = test_generator.classes
    y_pred_test = np.argmax(model.predict(test_generator), axis=1)
    test_cm = confusion_matrix(y_true_test, y_pred_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Training Confusion Matrix')

    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Testing Confusion Matrix')

    plt.show()

# Main function
def main(epochs, batch_size):
    train_generator, validation_generator, test_generator = load_data(batch_size)

    model = build_model()
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    plot_training_history(history)
    plot_confusion_matrices(model, train_generator, test_generator)

    # Ensure model directory exists
    os.makedirs("models", exist_ok=True)
    model.save("models/mnist_cnn_model.h5")
    print("Model saved successfully in 'models/mnist_cnn_model.h5'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    main(args.epochs, args.batch_size)
