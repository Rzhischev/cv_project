import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os, json, cv2, random

# Load the YOLO model
MODEL_PATH = './pages/pizza.pt'
model = YOLO(MODEL_PATH)

def app():
    st.title("Модель для обнаружения пицц")
    st.write("Эта страница использует YOLO модель для обнаружения пицц на изображениях.")
    
    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Read the image from uploaded file
        image = Image.open(uploaded_image)
        
        # Display the original image
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        
        # Perform inference with YOLO model
        results = model(image, conf=0.4)

        # Show the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            st.image(im, caption='Результат обнаружения', use_column_width=True)

# Run the Streamlit app
if __name__ == "__main__":
    app()

