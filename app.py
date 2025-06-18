import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

# Image dimensions for VGG-16 (224x224)
img_height = 224
img_width = 224

# Your class names
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 
            'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 
            'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 
            'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 
            'onion', 'orange', 'paprika', 'pear', 'peas', 
            'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 
            'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Load your VGG-16 based model
model = load_model(r'D:\computerVision\vgg16 model\vgg16_fruit_vegetable_model_final.keras')

def load_and_preprocess_image(image_file):
    # Load image
    #img = Image.open(image_file)
    img = Image.open(image_file).convert('RGB')
    img = img.resize((img_height, img_width))
    
    # Convert to array
    img_array = tf.keras.utils.img_to_array(img)
    
    # Expand dimensions
    img_array = tf.expand_dims(img_array, axis=0)
    
    # Preprocess for VGG-16
    img_array = preprocess_input(img_array)
    
    return img_array

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Model Prediction"])

# Main page
if app_mode == "Home":
    st.title("Welcome To Fruit And Vegetable Recognition System (PMAS AAUR)")
    image_path = r"D:\computerVision\download (1).jpg"
    st.image(image_path ,use_container_width=True )
    st.text("Select the page from the Sidebar")
    st.subheader("Streamlit App Explanation for Fruit and Vegetable Classification")

    st.text("This Streamlit application is designed for the automatic classification of \n"
            "fruit and vegetable images using a pre-trained deep learning model. The objective \n"
            "is to provide an efficient, user-friendly interface for identifying one of 36 fruit or \n"
            "vegetable categories from an uploaded image.")
       
elif app_mode == "About Project":
    st.title("About Project")
    st.subheader("About Dataset")
    st.text("This Dataset contains 3825 images of Fruits & Vegetable of following 36 category")
    st.code("Fruit Category: -> apple, banana, grapes, kiwi, lemon,\n"
        "mango, orange, pear, pineapple,"
        "pomegranate, watermelon\n",)
    st.code("Vegetable Category: -> beetroot, bell pepper, cabbage, "
        "capsicum, carrot,\ncauliflower, chilli pepper, corn, "
        "cucumber, eggplant, garlic, ginger,\n"
        "jalepeno, lettuce, onion, paprika, peas,"
        "potato, raddish, soy beans,\nspinach,"
        "sweetcorn, sweetpotato, tomato, turnip")
    st.subheader("Content")
    st.text("Dataset contain Three folders")
    st.text("1. Train :3115 files belonging to 36 classes.")
    st.text("2. Test  :351 files belonging to 36 classes.")
    st.text("3. Validation :359 files belonging to 36 classes.")
    st.subheader("About Model")

    st.text("The app is backed by a Convolutional Neural Network (CNN) model based on VGG16, \n"
        "a proven deep architecture originally developed for large-scale image classification tasks. \n"
        "The choice of VGG16 is motivated by its strong feature extraction capabilities, as it uses \n"
        "multiple layers of convolution and pooling to capture intricate visual patterns, such as texture, \n"
        "shape, and color—features crucial for distinguishing among visually similar items like \n"
        "fruits and vegetables.")

    st.text("The underlying model was trained using a structured dataset of 3825 images, divided into \n"
        "training, testing, and validation sets. Training was conducted using TensorFlow's high-level APIs, \n"
        "with preprocessing steps including resizing, normalization, and augmentation for robustness. \n"
        "After training, the model was saved and integrated into this Streamlit app for real-time predictions.")

    st.text("In the app:\n"
        "- Users can upload an image via the 'Model Prediction' section.\n"
        "- The image is resized to 224x224 pixels to fit VGG16’s input requirement.\n"
        "- It’s preprocessed using preprocess_input() specific to VGG16.\n"
        "- The model then predicts the most probable class, and the app displays the \n"
        "  predicted fruit or vegetable name with a confidence score.")

    st.text("This system is ideal for educational demonstrations, smart farming tools, \n"
        "food inventory apps, and beginner-level computer vision showcases. It encapsulates \n"
        "a full machine learning pipeline—from dataset handling and model training to deployment \n"
        "through an interactive web interface.")
    st.subheader("About App")
    st.text("This Streamlit application is designed for the automatic classification of \n"
            "fruit and vegetable images using a pre-trained deep learning model. The objective \n"
            "is to provide an efficient, user-friendly interface for identifying one of 36 fruit or \n"
            "vegetable categories from an uploaded image.")
    
elif app_mode == "Model Prediction":
    st.title("Model Prediction")
    
    test_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Predict"):
            # Process the image
            img_array = load_and_preprocess_image(test_image)
            
            # Make prediction
            predictions = model.predict(img_array)
            #score = tf.nn.softmax(predictions[0])
            score = predictions[0]
            # Get results
            predicted_class = data_cat[np.argmax(score)]
            confidence = np.max(score) * 100
            
            st.success(f"Model prediction: the image is {predicted_class} with an accuracy of {confidence:.2f}%")