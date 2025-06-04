import streamlit as st
import pandas as pd
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
import google.generativeai as genai
import os  # for file path handling
import openpyxl
import tensorflow as tf  # Add this import for TensorFlow
from keras.models import Model

# Initialize Gemini API
GEMINI_API_KEY = "AIzaSyDoaGwqLnuiB9bV26jm5gNLum7DTTYQM1M"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Load the trained model
model = load_model(r'C:\Users\satwi\Downloads\best_model.h5')  # Update path
labels = ['Erosive Gastritis', 'Peptic Ulcer', 'Gastroesophageal Reflux Disease (GERD)', 'Normal']  # Actual disease names

# Feedback storage file
feedback_file = r"C:\Users\satwi\Downloads\feedback_data.xlsx"

# Functions
def preprocess_image(image):
    image = img_to_array(image)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def generate_gradcam(model, image_array, class_index, layer_name="conv5_block3_out"):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(guided_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

def overlay_heatmap(heatmap, original_image, alpha=0.5):
    heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(original_image), 1-alpha, heatmap, alpha, 0)
    return Image.fromarray(superimposed_img)

# Streamlit App
st.title("Endoscopy Disease Detection ")

tab1, tab2, tab3 = st.tabs(["Disease Detection", "Chatbot Assistance", "Feedback"])

# Tab 1: Disease Detection
with tab1:
    st.header("Upload an Endoscopy Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)
        
        # Predict the class
        prediction = model.predict(processed_image)
        predicted_class = labels[np.argmax(prediction)]
        st.write(f"Prediction: *{predicted_class}*")
        
        # Grad-CAM Visualization
        st.subheader("Output")
        class_index = np.argmax(prediction)
        heatmap = generate_gradcam(model, processed_image, class_index)
        heatmap_overlay = overlay_heatmap(heatmap, image)

        # Create columns to display images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", width=300)  # Resize original image to 6x6 (approx)

        with col2:
            st.image(heatmap_overlay, caption="Grad-CAM Visualization", width=300)  # Resize Grad-CAM image to 6x6 (approx)
        
        # Display the prediction again below the images
        st.write(f"Prediction: *{predicted_class}*")

        if predicted_class != "Normal":
            ai_prompt = f"Provide a detailed description of {predicted_class}, including causes, symptoms, precautions, and treatments."
            ai_response = get_gemini_response(ai_prompt)
            st.write(ai_response)
            st.warning("Advice: Consult a gastroenterologist or specialist for a professional opinion.")

# Tab 2: Chatbot Assistance
with tab2:
    st.header("Ask the Chatbot")
    user_query = st.text_area("Describe your concerns or ask a question:")
    if st.button("Get Advice"):
        if user_query:
            ai_prompt = f"The user has the following concern: '{user_query}'. Provide helpful advice, including possible causes, suggestions, and general recommendations."
            chatbot_response = get_gemini_response(ai_prompt)
            st.write(chatbot_response)
        else:
            st.warning("Please enter your question or concern.")

# Tab 3: Feedback
with tab3:
    st.header("Submit Feedback")
    
    # Collect name, mobile number, email, and feedback
    name = st.text_input("Your Name:")
    mobile = st.text_input("Your Mobile Number:")
    email = st.text_input("Your Email:")
    feedback = st.text_area("We value your feedback:")
    
    if st.button("Submit Feedback"):
        if name and mobile and email and feedback:
            feedback_data = pd.DataFrame({'Name': [name], 'Mobile': [mobile], 'Email': [email], 'Feedback': [feedback]})
            
            # Ensure the feedback file path exists, create directory if necessary
            feedback_dir = os.path.dirname(feedback_file)
            if not os.path.exists(feedback_dir):
                os.makedirs(feedback_dir)
            
            # If the feedback file doesn't exist, create it, otherwise append
            if not os.path.exists(feedback_file):
                feedback_data.to_excel(feedback_file, index=False)
            else:
                existing_feedback = pd.read_excel(feedback_file)
                updated_feedback = pd.concat([existing_feedback, feedback_data], ignore_index=True)
                updated_feedback.to_excel(feedback_file, index=False)
            
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter your name, mobile number, email, and feedback before submitting.")

st.warning("This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
