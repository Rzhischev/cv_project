import streamlit as st

st.set_page_config(
    page_title= 'Computer Vision | ПРОЕКТ',
    page_icon="🤖",
    layout='wide'
    
)

st.sidebar.header("Home page")
c1, c2 = st.columns(2)

c2.image('https://kartinkof.club/uploads/posts/2023-05/1683610960_kartinkof-club-p-vetryak-kartinki-44.jpg')

c1.markdown("""
## Компьютерное зрение • Computer Vision
### Проект:
 ##### 1. Детекция ингридиентов пиццы с помощью YOLOv8
 ##### 2. Детекция ветряков с помощью YOLOv8
 ##### 3. Очищение документов от шумов с помощью автоэнкодера 
""")
