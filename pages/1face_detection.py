import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageFilter
import numpy as np
from io import BytesIO
import requests
import matplotlib.pyplot as plt 
import pandas as pd

#yolo12

model_path = "models/best.pt"

# Функция размытия
def blur_faces(image_pil, model_path=model_path):
    model = YOLO(model_path)
    image_np = np.array(image_pil)

    results = model(image_np, conf=0.5)

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                face = image_pil.crop((x1, y1, x2, y2))
                blurred_face = face.filter(ImageFilter.GaussianBlur(radius=20))
                image_pil.paste(blurred_face, (x1, y1, x2, y2))
    return image_pil

# Streamlit интерфейс
st.title("Детекция и блюр лиц")

# Выбор источника
option = st.radio("Выберите источник:", ["Загрузить файл", "Ввести ссылки (по одной в строке)"])

if option == "Загрузить файл":
    uploaded_files = st.file_uploader(
        "Загрузите одно или несколько изображений",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("🔍 Обработать файл(ы)"):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")

                # Показываем оригинал
                st.image(image, caption=f"Оригинал - {uploaded_file.name}", use_container_width=True)

                # Обрабатываем
                result_image = blur_faces(image.copy(), model_path=model_path)
                st.image(result_image, caption=f"Результат для {uploaded_file.name}", use_container_width=True)

elif option == "Ввести ссылки (по одной в строке)":
    urls_text = st.text_area("Введите ссылки, по одной в каждой строке:")
    if urls_text:
        if st.button("🔍 Обработать ссылку(и)"):
            urls = [url.strip() for url in urls_text.strip().splitlines() if url.strip()]
            for idx, url in enumerate(urls, 1):
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")

                    # Показываем оригинал
                    st.image(image, caption=f"Оригинал - URL {idx}", use_container_width=True)

                    # Обрабатываем
                    result_image = blur_faces(image.copy(), model_path=model_path)
                    st.image(result_image, caption=f"Результат для URL {idx}", use_container_width=True)

                except Exception as e:
                    st.error(f"Ошибка с URL {url}: {e}")

