import tensorflow as tf
import numpy as np
import pandas as pd
from utils import mnist_image, mnist_padded_image, fft_image, process_and_display_row  # Import existing functions

# Load trained model
model_path = "models/mnist_cnn_model.h5"
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Load dataset
df_test = pd.read_csv("mnist_test.csv")
df_test_labels = df_test['label']
df_test = df_test.drop('label', axis=1)

def preprocess_row(row):
    """Preprocesses an MNIST dataset row into a spectrogram for CNN input."""
    row_as_image = mnist_image(row)  # Convert row to 28x28 image
    row_as_image_padded = mnist_padded_image(row_as_image)  # Convert to 64x64 padded image
    row_as_image_fft = fft_image(row_as_image_padded)  # Convert to FFT spectrum
    
    # Normalize the spectrogram
    row_as_image_fft = row_as_image_fft.astype(np.float32) / np.max(row_as_image_fft)  # Scale between 0 and 1
    
    # Reshape for CNN input: (batch_size, 64, 64, 1)
    row_as_image_fft = np.expand_dims(row_as_image_fft, axis=[0, -1])  
    
    return row_as_image_fft

# Function to make a prediction on a given row index
def predict_row(index):
    """Predicts the digit for a row in the dataset."""
    row = df_test.iloc[index]  # Get the row data
    image = preprocess_row(row)  # Convert to CNN input format
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    print(prediction)

    confidence = np.max(prediction)
    
    print(f"Real: {df_test_labels[index]}")
    print(f"Predicted Digit: {predicted_label} (Confidence: {confidence:.2f})")
    
    # Display the row as images
    process_and_display_row(row)

# Example: Predict the first row
predict_row(0)
