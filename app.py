import streamlit as st
from pages import encoder_page, pizza_page, yolo8_page

# Define the pages in the app
PAGES = {
    "Encoder Page": encoder_page,
    "Pizza Page": pizza_page,
    "YOLO8 Page": yolo8_page
}

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Render the page selected by the user
    selected_page = PAGES[choice]
    selected_page.app()

# Run the app
if __name__ == "__main__":
    main()
