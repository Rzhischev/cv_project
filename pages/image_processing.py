from PIL import Image
import torchvision.transforms as transforms
import torch

def process_local_image(image_path, model, device):
    # 1. Загрузка изображения с диска
    image = Image.open(image_path).convert('L')

    # 2. Преобработка изображения
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # 3. Инференс (получение предсказания от модели)
    with torch.no_grad():
        output = model(image_tensor)

    return output
