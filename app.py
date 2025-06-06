import streamlit as st
from PIL import Image
import os

# Настройки страницы
st.set_page_config(
    page_title="Детекция и Сегментация",
    page_icon="📸",
    layout="wide"
)

# Заголовок
st.title("📸 Детекция и Сегментация")




# Подзаголовок с эмодзи
st.markdown("""
## 🚀 Добро пожаловать в приложение!


""")
# Добавляем раздел с кратким описанием
st.markdown("""
---

💡 Что умеет приложение?
- Загружайте одно или несколько изображений
- Обрабатывайте их с помощью выбранной модели
- Смотрите детекцию и сегментацию прямо в браузере

       """)   

st.markdown("### 👉 Здесь вы можете увидеть основные метрики по трём моделям")  


# Путь к папке с картинками
pictures_folder = "/Users/nikita/ds_bootcamp/Niyaz/phase-2/ds-phase-2/09-cv/proj_face_det/pictures"

# Определяем, какие картинки для какой модели
models = {
    "YOLOv12_medium": ["fd_total.jpg"],
    "YOLOv8_medium": ["turb1.jpg", "turb2.jpg", "turb3.jpg", "turb4.jpg"],
    "U-Net": ["seg.jpg"]
}

# Вывод по моделям
for model_name, picture_files in models.items():
    st.markdown(f"### {model_name}")
    cols = st.columns(2)  # 2 колонки для картинок

    for idx, file in enumerate(picture_files):
        image_path = os.path.join(pictures_folder, file)
        image = Image.open(image_path)
        with cols[idx % 2]:
            st.image(image, caption=file, use_container_width=True)
          


st.markdown("### 👉 Выберите модель слева, чтобы начать!")  


