import streamlit as st
from PIL import Image
import torch
from model import ConvAutoencoder
from image_processing import process_local_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models/encoding.pt'

def app():
    st.title("Модель Encoder")
    st.write("Эта страница использует модель Encoder для очистки изображений.")

    model = ConvAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        processed_image = process_local_image(uploaded_image, model, DEVICE)
        st.image(processed_image, caption='Обработанное изображение', use_column_width=True)
