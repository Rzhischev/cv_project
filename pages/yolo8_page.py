import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO

# Load the YOLO model
WEIGHTS_PATH = './pages/windmill1.pt'  # Path to your weights file
@st.cache(allow_output_mutation=True)
def load_model():
    """
    Load the YOLO model from the disk.
    The function is cached to reuse the loaded model across different runs of the Streamlit app,
    which saves time as loading the model can be expensive.
    """
    model = YOLO(WEIGHTS_PATH)
    return model

# Load the cached model
model = load_model()

def app():
    st.title("Модель обнаружения ветряков")
    st.write("Эта страница использует модель YOLOv8 для обнаружения ветряков на изображениях.")
    
    # Upload image through file uploader
    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    
    # Input box for user to enter image URL
    image_url = st.text_input("Или введите URL изображения", "")
    
    # Check if an image is uploaded
    if uploaded_image is not None:
        # Read the image from uploaded file
        image = Image.open(uploaded_image)
        # Process and display the image
        process_image(image)

    # Check if a URL is entered
    elif image_url:
        # Download the image from the URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        # Process and display the image
        process_image(image)
        
def process_image(image):
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


