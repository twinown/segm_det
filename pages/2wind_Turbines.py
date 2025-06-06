import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageFilter
import numpy as np
from io import BytesIO
import requests
import matplotlib.pyplot as plt 
import pandas as pd

#yolo8m

st.title("Детекция ветряных турбин")

# Выбор источника
option = st.radio("Выберите источник:", ["Загрузить файл", "Ввести ссылки (по одной в строке)"])

# Модель (замени на путь к своей модели)
model_path =  "/Users/nikita/ds_bootcamp/Niyaz/phase-2/ds-phase-2/09-cv/proj_face_det/models/best2.pt"
model = YOLO(model_path)

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

                # Детекция (без размытия)
                results = model(np.array(image), conf=0.5)
                for r in results:
                    annotated = r.plot()
                    st.image(annotated, caption=f"Результат для {uploaded_file.name}", use_container_width=True)
                    #st.write(r.boxes)

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

                    # Детекция (без размытия)
                    results = model(np.array(image), conf=0.5)
                    for r in results:
                        annotated = r.plot()

                        st.image(annotated, caption=f"Результат для URL {idx}", use_container_width=True)

                except Exception as e:
                    st.error(f"Ошибка с URL {url}: {e}")

