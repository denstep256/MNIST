import keras
import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from gtts import gTTS
import playsound
import os
from pyngrok import ngrok


def start_ngrok():
    public_url = ngrok.connect('8501')  # Streamlit по умолчанию работает на порту 8501
    st.write(f"Твоё приложение доступно по ссылке: {public_url}")

@st.cache_resource
def load_model():
    try:
        models = keras.models.load_model("/Users/denstep256/Documents/project/Python/AI/MNIST-AI/models/Model.keras")
        print("Model loaded successfully!")
        return models
    except Exception as ex:
        st.error(f"Error loading model: {ex}")
        st.stop()

model = load_model()

english_digits = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

russian_digits = {
    0: "Ноль", 1: "Один", 2: "Два", 3: "Три", 4: "Четыре",
    5: "Пять", 6: "Шесть", 7: "Семь", 8: "Восемь", 9: "Девять"
}


language_mappings = {
    "English": english_digits,
    "Русский": russian_digits,
}

st.title("Digit Recognition and Translation")
st.write("Draw a digit below:")

selected_language = st.selectbox("Select Language:", list(language_mappings.keys()))

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=[0, -1])
    return img

def translate_digit(digit, language):
    if 0 <= digit <= 9:
        return language_mappings[language][digit]
    else:
        return "Invalid Digit"

def speak(text, language):
    try:
        if language == "English":
            lang = 'en'
        elif language == "Русский":
            lang = 'ru'

        tts = gTTS(text=text, lang=lang, slow=False)
        filename = "temp_audio.mp3"
        tts.save(filename)
        st.audio(filename)
        os.remove(filename)
    except Exception as e:
        st.error(f"Error during speech synthesis: {e}")


if st.button("Predict"):
    try:
        if canvas_result.image_data is not None:
            img = np.array(canvas_result.image_data)
            if img.any():
                img = preprocess_image(img)
                prediction = model.predict(img)
                predicted_digit = np.argmax(prediction)
                translated_digit = translate_digit(predicted_digit, selected_language)
                st.write(f"Predicted Digit: {predicted_digit}")
                st.write(f"Translation ({selected_language}): {translated_digit}")
                speak(translated_digit, selected_language)

            else:
                st.write("Please draw a digit on the canvas.")
        else:
            st.write("Canvas data is not available.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
