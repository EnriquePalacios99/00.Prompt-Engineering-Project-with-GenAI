import streamlit as st
st.set_page_config(page_title="GenAI MVP", layout="wide")
st.title("GenAI MVP – Descripciones, Imágenes y Feedback")
st.markdown(
    "- **01_Descripciones**: genera copys listos para e-commerce.\n"
    "- **02_Imagenes**: produce creatividades promocionales a partir de un packshot.\n"
    "- **03_Feedback**: resume comentarios y clasifica sentimiento."
)
st.info("Configura tus credenciales en `.env` en la raíz del proyecto.")
