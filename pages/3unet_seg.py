import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import streamlit as st
from torchvision import transforms
from io import BytesIO
import requests

# Конфигурация страницы
st.set_page_config(page_title="Aerial Segmentation", page_icon="🛰", layout="wide")

# --- Модель U-Net ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        return self.final_conv(dec1)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/Users/nikita/ds_bootcamp/Niyaz/phase-2/ds-phase-2/09-cv/proj_face_det/models/aerial_forest_best.pt"
    
    try:
        model = torch.jit.load(model_path, map_location=device)
        return model.to(device).eval(), device
    except:
        try:
            model = UNet(in_channels=3, out_channels=2)
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            return model.to(device).eval(), device
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {str(e)}")
            return None, None

def preprocess_image(image, target_size=(512, 512)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, image.resize(target_size, Image.Resampling.LANCZOS)

def process_image(image_pil, model, device):
    input_tensor, resized_image = preprocess_image(image_pil)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        if len(output.shape) == 4:
            if output.shape[1] == 1:
                predictions = torch.sigmoid(output)
                mask = (predictions > 0.5).float().cpu().numpy()[0, 0]
            else:
                predictions = torch.softmax(output, dim=1)
                mask = torch.argmax(predictions, dim=1).cpu().numpy()[0]
        else:
            return None, None
    
    # Создаем цветную маску (лес - белый)
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_mask[mask == 1] = [255, 255, 255]
    
    return Image.fromarray(colored_mask), mask

# --- Функция для наложения маски без OpenCV ---
def overlay_mask_on_image_pil(original_image, mask, color=(0, 0, 255), alpha=0.5):
    original_np = np.array(original_image).copy()
    color_mask = np.zeros_like(original_np)
    color_mask[mask == 1] = color
    blended = original_np * (1 - alpha) + color_mask * alpha
    blended = blended.astype(np.uint8)
    return Image.fromarray(blended)

# --- Интерфейс ---
st.title("🛰 Сегментация аэроснимков")

option = st.radio("Выберите источник:", ["Загрузить файл", "Ввести ссылку"], horizontal=True)

if option == "Загрузить файл":
    uploaded_files = st.file_uploader(
        "Загрузите спутниковые снимки (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        model, device = load_model()
        if model:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                overlay, mask = process_image(image, model, device)
                
                if overlay is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Оригинал", use_container_width=True)
                    with col2:
                        st.image(overlay, caption="Сегментация (лес - белый)", use_container_width=True)
                    
                    # Третье изображение под ними
                    overlayed_image = overlay_mask_on_image_pil(
                        image.resize(overlay.size), mask, color=(0, 0, 255), alpha=0.5)
                    st.image(overlayed_image, caption="Оригинал + Сегментация (синий цвет)", use_container_width=True)

elif option == "Ввести ссылку":
    url = st.text_input("URL изображения:", placeholder="https://example.com/image.jpg")
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            model, device = load_model()
            if model:
                overlay, mask = process_image(image, model, device)
                
                if overlay is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Оригинал", use_container_width=True)
                    with col2:
                        st.image(overlay, caption="Сегментация (лес - белый)", use_container_width=True)
                    
                    # Третье изображение под ними
                    overlayed_image = overlay_mask_on_image_pil(
                        image.resize(overlay.size), mask, color=(0, 0, 255), alpha=0.5)
                    st.image(overlayed_image, caption="Оригинал + Сегментация (синий цвет)", use_container_width=True)
            
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")