import streamlit as st
from pages import encoder_page, pizza_page, yolo8_page
import ssl

# Отключение проверки SSL-сертификата
ssl._create_default_https_context = ssl._create_unverified_context

# Define the pages in the app
PAGES = {
    "Encoder Page": encoder_page,
    "Pizza Page": pizza_page,
    "YOLO8 Page": yolo8_page
}

st.set_page_config(
    page_title= 'Computer Vision | ПРОЕКТ',
    page_icon="🤖",
    layout='wide'
    
)
# Run the app
if __name__ == "__main__":
    main()
