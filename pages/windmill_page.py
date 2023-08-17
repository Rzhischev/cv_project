import streamlit as st
from PIL import Image
from ultralytics import YOLO

def app():
    st.title("Модель Windmill")
    st.write("Эта страница использует модель Windmill для обработки изображений.")
    
    # Загрузка модели
    model = YOLO('best.pt')
    
    # Загрузка изображения
    image_url = st.text_input('Введите URL изображения:', '')
    
    if image_url:
        # Обработка изображения
        results = model(image_url)
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            st.image(im, caption='Результат', use_column_width=True)
