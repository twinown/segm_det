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
pictures_folder = "pictures"

# Определяем, какие картинки для какой модели
models = {
    "YOLOv12_medium": ["ss.jpg"],
    "YOLOv8_medium": ["turb1.jpg", "turb2.jpg", "turb3.jpg", "turb4.jpg"],
    "U-Net": ["seg.jpg"]
}

# Вывод по моделям
for model_name, picture_files in models.items():
    st.markdown(f"### {model_name}")
    cols = st.columns(2)  # 2 колонки для картинок

    for idx, file in enumerate(picture_files):
        image_path = os.path.join(pictures_folder, file)
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            with cols[idx % 2]:
                st.image(image_bytes, caption=file, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка при загрузке {file}: {e}")
          


st.markdown("### 👉 Выберите модель слева, чтобы начать!")  


