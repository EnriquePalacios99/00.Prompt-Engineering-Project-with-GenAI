import streamlit as st
from services.llm_gemini import generate_product_description_gemini

st.title("Generación de descripciones (Gemini)")
name = st.text_input("Nombre del producto", "Cereales Ángel")
attrs = st.text_area("Atributos (texto/JSON breve)", "fortificado, crujiente, familiar")
channel = st.selectbox("Canal", ["ecommerce", "marketplace", "redes"])
images = st.file_uploader("Imágenes del producto (opcional)", type=["jpg","jpeg","png"], accept_multiple_files=True)

if st.button("Generar", type="primary"):
    img_bytes = [f.read() for f in (images or [])]
    out = generate_product_description_gemini(name, attrs, channel, img_bytes)
    st.subheader("Descripción corta"); st.write(out.get("short",""))
    st.subheader("Descripción larga (SEO)"); st.write(out.get("long",""))
    st.subheader("Bullets"); st.write(out.get("bullets", []))
    st.subheader("Hashtags"); st.write(out.get("hashtags", []))
    with st.expander("Salida completa (raw)"):
        st.code(out.get("raw",""), language="json")
