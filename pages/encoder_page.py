import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True)  # Bottleneck

        # decoder
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SELU()
        )
        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, bias=False),
            nn.BatchNorm2d(16),
            nn.SELU()
        )
        self.conv3_t = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.drop_out = nn.Dropout(0.5)

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.drop_out(x)
        x, indicies = self.pool(x)
        return x, indicies

    def decode(self, x, indicies):
        x = self.unpool(x, indicies)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        x = self.drop_out(x)
        x = self.conv3_t(x)
        return x

    def forward(self, x):
        latent, indicies = self.encode(x)
        out = self.decode(latent, indicies)
        return out

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = '/Users/rzhishchev/Downloads/encoding.pt'

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def process_image(image_path, model, device):
    image = Image.open(image_path).convert('L')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return output

def app():
    st.title("Модель Encoder")
    st.write("Эта страница использует модель Encoder для очистки изображений.")
    
    model = load_model(MODEL_PATH, DEVICE)
    
    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        
        processed_image = process_image(uploaded_image, model, DEVICE)
        
        # Преобразование тензора PyTorch в массив NumPy
        processed_image_numpy = processed_image.cpu().squeeze().numpy()
        
        # Преобразование массива NumPy в объект PIL Image
        processed_image_pil = Image.fromarray((processed_image_numpy * 255).astype('uint8'))
        
        # Показать обработанное изображение с помощью Streamlit
        st.image(processed_image_pil, caption='Обработанное изображение', use_column_width=True)
        
# Запустите Streamlit App
if __name__ == "__main__":
    app()
