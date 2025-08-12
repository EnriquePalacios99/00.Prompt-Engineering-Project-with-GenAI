import streamlit as st
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from services.images_gemini import generate_promos_with_gemini_background

st.set_page_config(page_title="Imágenes promocionales", page_icon="🖼️", layout="wide")
st.title("🖼️ Generador de imágenes promocionales (Vertex AI)")

with st.sidebar:
    st.subheader("Opciones")
    formato = st.selectbox("Formato", ["1080x1350 (IG Feed)", "1200x628 (Ads)"])
    n = st.number_input("N° de imágenes", min_value=1, max_value=4, value=1, step=1)
    brand_hex = st.color_picker("Color principal", "#E30613")
    bg_prompt = st.text_area(
        "Prompt del fondo (Vertex Imagen 3)",
        value="Fondo fotográfico limpio estilo estudio con luz suave y textura sutil, espacio negativo a la izquierda para texto.",
        height=100
    )

col1, col2 = st.columns([1, 1])
with col1:
    base = st.file_uploader("Sube packshot (PNG/JPG)", type=["png", "jpg", "jpeg"])
with col2:
    headline = st.text_input("Titular", "Nuevo Cereales Ángel")
    subheadline = st.text_input("Subtítulo", "Más sabor para tus mañanas")
    cta = st.text_input("CTA", "Compra ahora")

size = (1080, 1350) if "1080x1350" in formato else (1200, 628)

def _to_png_bytes(img_arr):
    bio = BytesIO()
    Image.fromarray(img_arr).save(bio, format="PNG")
    bio.seek(0)
    return bio

st.divider()
generate = st.button("Generar con Vertex AI", type="primary")

if generate:
    if not base:
        st.warning("Sube un packshot para continuar.")
        st.stop()

    # Validar que el archivo subido es imagen válida
    try:
        _ = Image.open(base).verify()
        base.seek(0)  # volver al inicio para leer bytes luego
    except UnidentifiedImageError:
        st.error("El archivo subido no es una imagen válida. Intenta con PNG/JPG.")
        st.stop()

    try:
        with st.spinner("Generando creatividades con Vertex Imagen 3..."):
            imgs = generate_promos_with_gemini_background(
                base_bytes=base.read(),
                headline=headline,
                subheadline=subheadline,
                cta=cta,
                n=int(n),
                canvas_size=size,
                brand_hex=brand_hex,
                bg_prompt=bg_prompt,
            )

        st.success(f"Listo. Se generaron {len(imgs)} creatividad(es) con Vertex AI.")
        for i, arr in enumerate(imgs, 1):
            st.image(arr, caption=f"Creatividad {i}", use_container_width=True)
            st.download_button(
                label=f"Descargar PNG {i}",
                data=_to_png_bytes(arr),
                file_name=f"creatividad_{i}.png",
                mime="image/png",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Ocurrió un error generando con Vertex AI: {e}")
        st.info(
            "Verifica que tu `.env` tenga GCP_PROJECT, GCP_LOCATION y GOOGLE_APPLICATION_CREDENTIALS, "
            "que el modelo `imagen-3.0-generate-001` esté disponible en tu región y que el service account "
            "tenga el rol `roles/aiplatform.user`."
        )
