import streamlit as st
import windmill_page
import pizza_page
import encoder_page

PAGES = {
    "Модель Windmill": windmill_page,
    "Модель Pizza": pizza_page,
    "Модель Encoder": encoder_page
}

st.sidebar.title('Навигация')
selection = st.sidebar.radio("Перейти к странице:", list(PAGES.keys()))
page = PAGES[selection]
page.app()
