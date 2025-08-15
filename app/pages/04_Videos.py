# app/pages/04_Videos.py
import streamlit as st
from io import BytesIO
from services.video_veo import generate_promo_videos

st.set_page_config(page_title="Videos promocionales (Veo)", page_icon="🎬", layout="wide")
st.title("🎬 Generador de videos promocionales (Veo)")

with st.sidebar:
    st.subheader("⚙️ Configuración")
    mode = st.radio(
        "Modo", ["Imagen → Video", "Solo texto → Video"], index=0,
        help="Usa 'Imagen → Video' para animar tu packshot con movimiento."
    )

    model = st.selectbox(
        "Modelo Veo",
        [
            "veo-3.0-fast-generate-001 (rápido, 720p)",
            "veo-3.0-generate-001 (HQ, 720p)",
            "veo-2.0-generate-001 (imagen→video GA)",
            "veo-3.0-generate-preview (imagen→video preview)",
        ],
        help=(
            "• Texto→Video: usa 'veo-3.0-fast-generate-001' o 'veo-3.0-generate-001'.\n"
            "• Imagen→Video: 'veo-2.0-generate-001' (GA) o 'veo-3.0-generate-preview' (preview)."
        ),
    )
    model_id = model.split(" ")[0]

    aspect = st.selectbox("Aspect ratio", ["16:9", "9:16", "1:1"], index=0)
    duration = st.slider("Duración (seg)", 2, 8, 8, help="Veo 3 produce hasta ~8 segundos por clip.")
    # Resolución referencial (el SDK Python actual suele devolver 720p por defecto)
    resolution = st.selectbox("Resolución (referencial)", ["720p"], index=0,
                              help="El SDK Python puede ignorar este campo; Veo 3 devuelve 720p por defecto.")
    number = st.slider("N° de videos", 1, 2, 1)
    gen_audio = st.toggle("Generar audio (si está disponible)", value=False,
                          help="Puede que tu despliegue no soporte audio; si falla, desactívalo.")

    st.divider()
    st.caption("💡 Recomendaciones de prompt:")
    st.markdown(
        "- Describe **movimiento de cámara** (paneo, acercamiento suave).\n"
        "- Indica **iluminación** y **escena** (estudio, fondo limpio, mesa de madera, etc.).\n"
        "- Evita pedir **texto en pantalla** o **logos**."
    )

col1, col2 = st.columns([1, 1])

with col1:
    if mode == "Imagen → Video":
        pack = st.file_uploader("Sube packshot (PNG/JPG)", type=["png", "jpg", "jpeg"])
    else:
        pack = None

    product = st.text_input("Producto", "Cereales Ángel")
    brand = st.text_input("Marca", "Ángel")

with col2:
    base_prompt = st.text_area(
        "Prompt principal",
        value="Packshot del producto de cereal de maíz en un estudio. " \
        "El fondo es un campo de plantas de maíz en movimiento sutil y continuo. " \
        "El movimiento de cámara es un paneo lateral combinado con un leve dolly-in, grabado a 24 fps. " \
        "La iluminación incluye una luz principal a 45°, una luz de relleno baja para suavizar sombras y una tenue rim light para separar el producto del fondo. " \
        "El enfoque principal está en enfatizar el logo del producto y la textura del empaque."
,
        height=120,
    )
    style_hint = st.text_input("Estilo (opcional)", "Estudio, moderno, colores cálidos")
    negative = st.text_input("Negative prompt (opcional)", "texto en pantalla, logotipos inventados, manos, personas")

st.divider()
go = st.button("🚀 Generar video", type="primary", use_container_width=True)

def _to_downloadable(mp4_bytes: bytes, idx: int) -> BytesIO:
    bio = BytesIO()
    bio.write(mp4_bytes)
    bio.seek(0)
    return bio

if go:
    if mode == "Imagen → Video" and not pack:
        st.warning("Sube un packshot para continuar."); st.stop()

    image_bytes = pack.read() if pack else None

    # Tips de compatibilidad modelo ↔ modo
    if mode == "Imagen → Video" and model_id.startswith("veo-3.0") and "preview" not in model_id:
        st.info("Sugerencia: para imagen→video usa 'veo-2.0-generate-001' o 'veo-3.0-generate-preview' (según tu región).")
    if mode == "Solo texto → Video" and model_id.startswith("veo-2.0"):
        st.info("Sugerencia: para texto→video usa 'veo-3.0-fast-generate-001' o 'veo-3.0-generate-001'.")

    with st.spinner("Generando con Veo (puede tardar algunos minutos)…"):
        try:
            results = generate_promo_videos(
                prompt=base_prompt,
                negative_prompt=negative,
                product_image_bytes=image_bytes,
                model=model_id,
                aspect_ratio=aspect,
                duration_seconds=int(duration),
                resolution=resolution,             # informativo: el servicio lo registra pero no lo manda al SDK
                number_of_videos=int(number),
                generate_audio=bool(gen_audio),
                brand=brand,
                product_name=product,
                style_hint=style_hint,
            )
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
            st.stop()

    st.success(f"Listo. Se generó {len(results)} video(s).")

    for i, r in enumerate(results, 1):
        st.markdown(f"### Resultado {i}")
        mp4 = r.get("video_bytes")
        gcs_uri = r.get("gcs_uri")

        if mp4:
            st.video(mp4)
            st.download_button(
                label=f"⬇️ Descargar MP4 {i}",
                data=_to_downloadable(mp4, i),
                file_name=f"promo_{i}.mp4",
                mime=r.get("mime_type", "video/mp4"),
                use_container_width=True,
            )
        elif gcs_uri:
            st.warning("El SDK no devolvió bytes inline. Guardamos en GCS (revisa tu bucket).")
            st.code(gcs_uri)
        else:
            st.error("No se pudo obtener el video generado (sin bytes ni URI). Revisa el modelo y permisos.")
