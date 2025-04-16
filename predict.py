import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = tf.keras.models.load_model('potato_disease_model.h5')


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array


def predict_disease(img_path):
    img_array = preprocess_image(img_path)  # Preprocess the image
    prediction = model.predict(img_array)  # Predict using the model
    
    
    class_names = ['Healthy', 'Early Blight', 'Late Blight']
    predicted_class = class_names[np.argmax(prediction)]  # Get the class with highest probability
    confidence = np.max(prediction) * 100  # Confidence percentage

    return predicted_class, confidence

# Main function to run the prediction
if __name__ == "__main__":
    img_path = input("Enter the path of your potato leaf image: ")
    
    # Check if the file exists
    if os.path.exists(img_path):
        disease, confidence = predict_disease(img_path)  # Get the prediction
        print(f"Prediction: {disease}")
        print(f"Confidence: {confidence:.2f}%")
    else:
        print("The file path is invalid. Please check the image path and try again.")

