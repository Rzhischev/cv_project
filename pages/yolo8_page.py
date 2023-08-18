import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
# CONFIG_PATH = 'yolov8n-seg.yaml'  # Path to your YAML configuration file
WEIGHTS_PATH = 'windmill.pt'  # Path to your weights file
# model = YOLO(CONFIG_PATH)
model = YOLO(WEIGHTS_PATH)

def app():
    st.title("Модель для обнаружения ветряков")
    st.write("Эта страница использует YOLO модель для обнаружения ветряков на изображениях.")
    
    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Read the image from uploaded file
        image = Image.open(uploaded_image)
        
        # Display the original image
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        
        # Perform inference with YOLO model
        results = model(image)

        # Show the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            st.image(im, caption='Результат обнаружения', use_column_width=True)

# Run the Streamlit app
if __name__ == "__main__":
    app()

