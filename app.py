import streamlit as st

st.set_page_config(
    page_title= 'Computer Vision | ПРОЕКТ',
    page_icon="🤖",
    layout='wide'
    
)

st.sidebar.header("Home page")
c1, c2 = st.columns(2)

c2.image('https://s41256.pcdn.co/wp-content/uploads/2019/04/SLIDER-Appen_image_annotation_05.jpg')

c1.markdown("""
# Проект по Компьютерному зрению
### Cостоит из 3 частей:
 ##### 1. Детекция ингридиентов пиццы с помощью YOLOv5
 ##### 2. Детекция опухулей мозга по фотографии с помощью YOLOv5
 ##### 3. Очищение документов от шумов с помощью автоэнкодера 
""")
