import streamlit as st
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from services.images_gemini import generate_promos_with_gemini_background

st.set_page_config(page_title="Imágenes promocionales", page_icon="🖼️", layout="wide")
st.title("🖼️ Generador de imágenes promocionales (Vertex AI)")

with st.sidebar:
    st.subheader("Opciones generales")
    formato = st.selectbox("Formato", ["1080x1350 (IG Feed)", "1200x628 (Ads)"])
    n = st.number_input("N° de imágenes", min_value=1, max_value=4, value=1, step=1)
    brand_hex = st.color_picker("Color de marca (influye en el prompt)", "#E30613")
    bg_prompt = st.text_area(
        "Prompt del fondo (Vertex Imagen 3)",
        value="Una ilustración de caricatura caprichosa y vibrante ambientada en una cocina soleada, con una escena lúdica con hojuelas de maíz antropomórficas, salpicaduras de leche dinámicas, un tazón de cereal y una cuchara. El fondo tiene suaves degradados de amarillo brillante y azul claro, lo que sugiere un ambiente alegre. Numerosas hojuelas de maíz grandes y sonrientes, cada una con ojos anchos y expresivos, mejillas sonrosadas y pequeños brazos y piernas, están esparcidas por la escena. Algunas hojuelas flotan en el aire con los brazos saludando, mientras que otras están cerca de un tazón de cereal central. Salpicaduras dinámicas de leche blanca se congelan en movimiento, formando arcos y remolinos alrededor de los copos de maíz y el tazón, con varias gotas de leche suspendidas en el aire. El foco central es un alegre tazón a rayas azules y blancas rebosante de hojuelas de maíz doradas y un remolino de leche. Una cuchara metálica, representada con un brillo de caricatura, está parcialmente sumergida en el cereal, a punto de servir. La paleta de colores es brillante y acogedora, dominada por amarillos cálidos, blancos cremosos y azules fríos, acentuados con toques de naranja y rojo. Las líneas son limpias y audaces, características de la animación infantil, y la escena se representa con un brillo suave y acogedor.",
        height=100
    )

    st.subheader("Textos")
    headline_hex = st.color_picker("Color del Titular", "#141414")
    subheadline_hex = st.color_picker("Color del Subtítulo", "#3C3C3C")
    cta_hex = st.color_picker("Color del texto del CTA", "#E30613")

    st.subheader("Cuarto de circunferencia (placa)")
    quarter_radius_pct = st.slider("Tamaño del cuarto (%)", 20, 95, 55, help="Porcentaje del lado menor del canvas.")
    plate_hex = st.color_picker("Color de la placa", "#FFFFFF")
    plate_opacity = st.slider("Opacidad de la placa", 0, 255, 180)

    st.subheader("Rayos (desde esquina inferior derecha)")
    rays_enabled = st.checkbox("Activar rayos", value=True)
    rays_count = st.slider("Cantidad de rayos", 0, 40, 12)
    rays_length_pct = st.slider("Longitud relativa (× lado menor)", 5, 150, 60)
    rays_thickness_px = st.slider("Grosor (px)", 1, 25, 6)
    rays_color_hex = st.color_picker("Color de rayos", "#FFD700")
    rays_opacity = st.slider("Opacidad de rayos", 0, 255, 180)
    rays_spread_deg = st.slider("Apertura (grados)", 0, 180, 80)

    st.subheader("Packshot")
    pack_scale_pct = st.slider("Tamaño del packshot (%)", 20, 200, 90)
    margin_right_pct = st.slider("Margen derecho (%)", 0, 50, 6)
    margin_bottom_pct = st.slider("Margen inferior (%)", 0, 50, 6)

    st.subheader("Sombra de base")
    shadow_scale_x = st.slider("Ancho relativo de sombra (%)", 30, 200, 90)
    shadow_scale_y = st.slider("Alto relativo de sombra (%)", 2, 50, 8)
    shadow_offset_y_px = st.slider("Desplazamiento vertical (px)", 0, 40, 6)
    shadow_opacity = st.slider("Opacidad", 0, 255, 160)
    shadow_blur_px = st.slider("Difuminado (px)", 0, 40, 12)

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
                headline_hex=headline_hex,
                subheadline_hex=subheadline_hex,
                cta_hex=cta_hex,
                quarter_radius_pct=quarter_radius_pct / 100.0,
                plate_hex=plate_hex,
                plate_opacity=int(plate_opacity),
                rays_enabled=bool(rays_enabled),
                rays_count=int(rays_count),
                rays_length_pct=rays_length_pct / 100.0,
                rays_thickness_px=int(rays_thickness_px),
                rays_color_hex=rays_color_hex,
                rays_opacity=int(rays_opacity),
                rays_spread_deg=float(rays_spread_deg),
                pack_scale_pct=pack_scale_pct / 100.0,
                margin_right_pct=margin_right_pct / 100.0,
                margin_bottom_pct=margin_bottom_pct / 100.0,
                shadow_scale_x=shadow_scale_x / 100.0,
                shadow_scale_y=shadow_scale_y / 100.0,
                shadow_offset_y_px=int(shadow_offset_y_px),
                shadow_opacity=int(shadow_opacity),
                shadow_blur_px=int(shadow_blur_px),
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
