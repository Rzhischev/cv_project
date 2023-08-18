import streamlit as st
from PIL import Image
from ultralytics import YOLO

def app():
    st.title("Модель Pizza")
    st.write("Эта страница использует модель Pizza для обработки изображений.")
    
    # Загрузка модели
    model = YOLO('-------best_pizza.pt')
    
    # Загрузка изображения
    image_url = st.text_input('Введите URL изображения:', '')
    
    if image_url:
        # Обработка изображения
        results = model(image_url, conf = 0.4)
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            st.image(im, caption='Результат', use_column_width=True)
