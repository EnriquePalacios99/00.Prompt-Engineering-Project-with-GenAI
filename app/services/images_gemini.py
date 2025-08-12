from typing import List, Tuple
import os
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from dotenv import load_dotenv
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel


# ==========================
# Utilidades de composición
# ==========================
def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _font(sz: int, bold: bool = True):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()

def _fit_shadow(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    img = img.convert("RGBA")
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    shadow = Image.new("RGBA", (img.width + 30, img.height + 30), (0, 0, 0, 0))
    s = Image.new("RGBA", (img.width, img.height), (0, 0, 0, 120))
    shadow.paste(s, (15, 15), s)
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    can = Image.new("RGBA", shadow.size, (0, 0, 0, 0))
    can.alpha_composite(shadow, (0, 0))
    can.alpha_composite(img, (0, 0))
    return can

def _draw_texts_and_cta(bg: Image.Image, headline: str, subheadline: str, cta: str, brand_rgb: tuple):
    W, H = bg.size
    d = ImageDraw.Draw(bg)

    fH = _font(int(H * 0.06), bold=True)
    fS = _font(int(H * 0.035), bold=False)
    fC = _font(int(H * 0.035), bold=False)

    x0, y0, text_w = int(W * 0.07), int(H * 0.18), int(W * 0.42)

    def draw_wrap(text, box, font, fill=(20, 20, 20)):
        x0b, y0b, x1b, _ = box
        max_w = x1b - x0b
        words = (text or "").split()
        cur, lines = "", []
        for w in words:
            t = (cur + " " + w).strip()
            if d.textlength(t, font=font) <= max_w:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        y = y0b
        lh = (font.size + 6 if hasattr(font, "size") else 22)
        for ln in lines:
            d.text((x0b, y), ln, font=font, fill=fill)
            y += lh
        return y

    y = draw_wrap(headline, (x0, y0, x0 + text_w, y0 + int(H * 0.18)), fH, (20, 20, 20))
    y += int(H * 0.02)
    y = draw_wrap(subheadline, (x0, y, x0 + text_w, y + int(H * 0.16)), fS, (60, 60, 60))
    y += int(H * 0.04)

    # CTA pill
    btn_w, btn_h = int(text_w * 0.75), int(H * 0.08)
    btn_x, btn_y = x0, y
    d.rounded_rectangle([btn_x, btn_y, btn_x + btn_w, btn_y + btn_h], radius=int(btn_h / 2), fill=(255, 255, 255, 230))
    tw = d.textlength(cta or "", font=fC)
    d.text(
        (btn_x + (btn_w - tw) / 2, btn_y + btn_h / 2 - (fC.size if hasattr(fC, "size") else 18) / 2),
        cta or "",
        font=fC,
        fill=brand_rgb
    )

def _compose_with_packshot(
    base_bytes: bytes,
    canvas_size: Tuple[int, int],
    headline: str,
    subheadline: str,
    cta: str,
    brand_hex: str,
    background_img: Image.Image
) -> np.ndarray:
    brand_rgb = _hex_to_rgb(brand_hex)
    W, H = canvas_size

    # Fondo (siempre viene de Vertex; redimensionamos al canvas)
    bg = background_img.convert("RGBA").resize((W, H), Image.LANCZOS)

    # Packshot + sombra
    prod = Image.open(BytesIO(base_bytes)).convert("RGBA")
    pack = _fit_shadow(prod, int(W * 0.6), int(H * 0.55))
    bg.alpha_composite(pack, (int(W * 0.55), int(H * 0.18)))

    # Textos
    _draw_texts_and_cta(bg, headline, subheadline, cta, brand_rgb)

    return np.array(bg.convert("RGB"))


# ==========================
# Generación con Vertex (Imagen 3)
# ==========================
def _vertex_generate_background(W: int, H: int, prompt: str, brand_hex: str) -> Image.Image:
    """
    Genera un fondo con Vertex Imagen 3 (usa tamaño por defecto del modelo y luego resize).
    Lanza excepción si algo falla (para que la UI lo muestre).
    """
    load_dotenv()
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    model_name = os.getenv("GEMINI_IMAGE_MODEL", "imagen-3.0-generate-001")
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not project or not location or not creds:
        raise RuntimeError("Faltan variables en .env: GCP_PROJECT, GCP_LOCATION o GOOGLE_APPLICATION_CREDENTIALS.")

    vertexai.init(project=project, location=location)
    model = ImageGenerationModel.from_pretrained(model_name)

    color_hint = f" Paleta acorde al color #{brand_hex.lstrip('#')}."
    full_prompt = (prompt or "").strip() + color_hint

    # No pasamos 'size' para evitar errores de soporte; el SDK genera (normalmente 1024x1024)
    gen = model.generate_images(
        prompt=full_prompt,
        number_of_images=1,
        safety_filter_level="block_few"
    )

    img_obj = gen.images[0]
    img_bytes = getattr(img_obj, "image_bytes", None) or getattr(img_obj, "_image_bytes", None)
    if not img_bytes:
        raise RuntimeError("No se obtuvieron bytes de imagen desde el SDK de Vertex.")

    bg = Image.open(BytesIO(img_bytes)).convert("RGB").resize((W, H), Image.LANCZOS)
    return bg


# ==========================
# API principal (para Streamlit)
# ==========================
def generate_promos_with_gemini_background(
    base_bytes: bytes,
    headline: str,
    subheadline: str,
    cta: str,
    n: int,
    canvas_size: Tuple[int, int],
    brand_hex: str,
    bg_prompt: str = "Fondo fotográfico limpio estilo estudio con luz suave y textura sutil, espacio negativo a la izquierda para texto."
) -> List[np.ndarray]:
    """
    Genera n creatividades usando SIEMPRE Vertex Imagen 3 para el fondo.
    Si Vertex falla, se propaga la excepción (para mostrarla en Streamlit).
    """
    W, H = canvas_size
    outs: List[np.ndarray] = []

    for _ in range(max(1, int(n))):
        bg_img = _vertex_generate_background(W, H, bg_prompt, brand_hex)
        arr = _compose_with_packshot(
            base_bytes=base_bytes,
            canvas_size=canvas_size,
            headline=headline,
            subheadline=subheadline,
            cta=cta,
            brand_hex=brand_hex,
            background_img=bg_img
        )
        outs.append(arr)

    return outs
