from typing import List, Tuple, Optional
import os
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

from dotenv import load_dotenv
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel


# ==========================
# Utilidades
# ==========================
def _hex_to_rgb(h: Optional[str]) -> tuple:
    h = (h or "").strip().lstrip("#")
    if len(h) != 6:
        return (20, 20, 20)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _font(sz: int, bold: bool = True):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()

def _fit_shadow(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Redimensiona con sombra suave alrededor (queda con halo)."""
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

def _quarter_circle_mask(W: int, H: int, R: int) -> Image.Image:
    """Máscara 'L' de un círculo centrado en (W, H) recortada al canvas: deja visible el cuarto inferior derecho."""
    mask_big = Image.new("L", (W + R, H + R), 0)
    d = ImageDraw.Draw(mask_big)
    d.ellipse((W - R, H - R, W + R, H + R), fill=255)
    return mask_big.crop((0, 0, W, H))

def _draw_texts_and_cta(
    bg: Image.Image,
    headline: str,
    subheadline: str,
    cta: str,
    headline_rgb: tuple,
    subheadline_rgb: tuple,
    cta_rgb: tuple
):
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

    y = draw_wrap(headline, (x0, y0, x0 + text_w, y0 + int(H * 0.18)), fH, headline_rgb)
    y += int(H * 0.02)
    y = draw_wrap(subheadline, (x0, y, x0 + text_w, y + int(H * 0.16)), fS, subheadline_rgb)
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
        fill=cta_rgb
    )

def _rays_layer(
    W: int, H: int,
    count: int,
    length_pct: float,
    thickness_px: int,
    spread_deg: float,
    color_hex: Optional[str],
    opacity: int
) -> Image.Image:
    """Crea una capa RGBA con 'rayos' saliendo desde la esquina inferior derecha (W,H)."""
    color = _hex_to_rgb(color_hex or "#FFD700")
    alpha = max(0, min(255, int(opacity)))
    length = int(min(W, H) * max(0.05, min(1.5, length_pct)))  # admite >100% si se desea
    thickness = max(1, int(thickness_px))
    spread = max(0.0, min(180.0, spread_deg))

    base_angle_deg = 225.0  # hacia arriba-izquierda
    half_spread = spread / 2.0

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    if count <= 0:
        return layer

    angles = []
    if count == 1:
        angles = [base_angle_deg]
    else:
        for i in range(count):
            t = i / (count - 1)  # 0..1
            angles.append(base_angle_deg - half_spread + t * spread)

    for ang in angles:
        rad = math.radians(ang)
        ex = int(W + length * math.cos(rad))
        ey = int(H + length * math.sin(rad))
        d.line([(W, H), (ex, ey)], fill=color + (alpha,), width=thickness)
    return layer

def _ground_shadow_layer(
    W: int, H: int,
    px: int, py: int, w: int, h: int,
    scale_x: float, scale_y: float,
    offset_y_px: int,
    opacity: int,
    blur_radius: int
) -> Image.Image:
    """Crea una elipse difuminada como sombra de base debajo del packshot."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    cx = px + w // 2
    cy = py + h + int(offset_y_px)

    ew = max(2, int(w * max(0.3, min(2.0, scale_x))))
    eh = max(2, int(h * max(0.02, min(0.5, scale_y))))

    x0 = cx - ew // 2
    y0 = cy - eh // 2
    x1 = cx + ew // 2
    y1 = cy + eh // 2

    alpha = max(0, min(255, int(opacity)))
    d.ellipse([x0, y0, x1, y1], fill=(0, 0, 0, alpha))

    blur = max(0, int(blur_radius))
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    return layer


# ==========================
# Composición principal
# ==========================
def _compose_with_packshot(
    base_bytes: bytes,
    canvas_size: Tuple[int, int],
    headline: str,
    subheadline: str,
    cta: str,
    headline_hex: str,
    subheadline_hex: str,
    cta_hex: str,
    background_img: Image.Image,
    # placa (cuarto de circunferencia)
    quarter_radius_pct: float,
    plate_hex: Optional[str],
    plate_opacity: int,
    # rayos
    rays_enabled: bool,
    rays_count: int,
    rays_length_pct: float,
    rays_thickness_px: int,
    rays_color_hex: Optional[str],
    rays_opacity: int,
    rays_spread_deg: float,
    # packshot
    pack_scale_pct: float,
    margin_right_pct: float,
    margin_bottom_pct: float,
    # sombra base
    shadow_scale_x: float,
    shadow_scale_y: float,
    shadow_offset_y_px: int,
    shadow_opacity: int,
    shadow_blur_px: int,
) -> np.ndarray:
    W, H = canvas_size
    headline_rgb = _hex_to_rgb(headline_hex)
    subheadline_rgb = _hex_to_rgb(subheadline_hex)
    cta_rgb = _hex_to_rgb(cta_hex)

    # Fondo de Vertex (capa base)
    bg = background_img.convert("RGBA").resize((W, H), Image.LANCZOS)

    # 1) Placa en cuarto de circunferencia (debajo de todo lo demás)
    quarter_radius_pct = max(0.2, min(0.95, quarter_radius_pct))
    R = int(min(W, H) * quarter_radius_pct)
    quarter_mask = _quarter_circle_mask(W, H, R)

    if plate_hex and plate_opacity > 0:
        plate_rgb = _hex_to_rgb(plate_hex)
        plate = Image.new("RGBA", (W, H), plate_rgb + (max(0, min(255, int(plate_opacity))),))
        bg.paste(plate, (0, 0), quarter_mask)

    # 2) Rayos (entre la placa y el packshot)
    if rays_enabled and rays_count > 0:
        rays = _rays_layer(
            W, H,
            count=int(rays_count),
            length_pct=float(rays_length_pct),
            thickness_px=int(rays_thickness_px),
            spread_deg=float(rays_spread_deg),
            color_hex=rays_color_hex,
            opacity=int(rays_opacity)
        )
        bg.alpha_composite(rays)

    # 3) Packshot (encima de la placa y rayos) con sombra SOLO en la base
    prod = Image.open(BytesIO(base_bytes)).convert("RGBA")
    pack_scale_pct = max(0.2, min(2.0, pack_scale_pct))  # 20% a 200% del tamaño base relativo a R
    target = int(R * 0.9 * pack_scale_pct)
    prod_fit = _fit_shadow(prod, target, target)  # mantiene halo, pero lo ocultaremos con sombra base

    # Posición
    margin_right = int(min(W, H) * max(0.0, min(0.5, margin_right_pct)))
    margin_bottom = int(min(W, H) * max(0.0, min(0.5, margin_bottom_pct)))
    px = W - margin_right - prod_fit.width
    py = H - margin_bottom - prod_fit.height

    # Sombra de base (elipse) — debajo del packshot
    shadow = _ground_shadow_layer(
        W, H,
        px, py, prod_fit.width, prod_fit.height,
        scale_x=shadow_scale_x,
        scale_y=shadow_scale_y,
        offset_y_px=shadow_offset_y_px,
        opacity=shadow_opacity,
        blur_radius=shadow_blur_px
    )
    bg.alpha_composite(shadow)

    # Pegamos el packshot encima
    bg.alpha_composite(prod_fit, (px, py))

    # 4) Textos al final
    _draw_texts_and_cta(bg, headline, subheadline, cta, headline_rgb, subheadline_rgb, cta_rgb)

    return np.array(bg.convert("RGB"))


# ==========================
# Generación con Vertex (Imagen 3)
# ==========================
def _vertex_generate_background(
    W: int,
    H: int,
    prompt: str,
    brand_hex: str,
    negative_prompt: Optional[str] = None
) -> Image.Image:
    """Genera un fondo con Vertex Imagen 3 (usa tamaño por defecto del modelo y luego resize)."""
    load_dotenv()
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    model_name = os.getenv("GEMINI_IMAGE_MODEL", "imagen-3.0-generate-001")
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not project or not location or not creds:
        raise RuntimeError("Faltan variables en .env: GCP_PROJECT, GCP_LOCATION o GOOGLE_APPLICATION_CREDENTIALS.")

    vertexai.init(project=project, location=location)
    model = ImageGenerationModel.from_pretrained(model_name)

    # Mejoramos el prompt: fotografía de estudio limpia y minimal, espacio negativo y soft light.
    brand_hint = f" Paleta coherente con el color #{(brand_hex or '').lstrip('#')}."
    base_prompt = (prompt or "").strip()
    full_prompt = (base_prompt + brand_hint).strip()

    # Intentamos usar negative_prompt si el SDK lo soporta; si no, fallback sin romper.
    try:
        gen = model.generate_images(
            prompt=full_prompt,
            number_of_images=1,
            safety_filter_level="block_few",
            negative_prompt=(negative_prompt or None)
        )
    except TypeError:
        # Algunas versiones no aceptan negative_prompt
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
# API principal
# ==========================
def generate_promos_with_gemini_background(
    base_bytes: bytes,
    headline: str,
    subheadline: str,
    cta: str,
    n: int,
    canvas_size: Tuple[int, int],
    brand_hex: str,
    # Prompt de fondo mejorado por defecto (fotografía de estudio minimalista)
    bg_prompt: str = (
        "Fondo fotográfico de estudio limpio y moderno, con gradiente sutil y textura ligera. "
        "Iluminación suave difusa (softbox a 45°), sombras muy suaves. "
        "Espacio negativo amplio en el lado izquierdo para texto. "
        "Estética minimalista y pulcra coherente con la marca."
    ),
    # Negative prompt opcional (para evitar overlays/personas/logo)
    bg_negative: Optional[str] = (
        "texto en pantalla, tipografías, logotipos inventados, marcas de agua, watermark, "
        "personas, manos, rostros, desorden, artefactos, glitch, ruido"
    ),
    headline_hex: str = "#141414",
    subheadline_hex: str = "#3C3C3C",
    cta_hex: str = "#E30613",
    # placa
    quarter_radius_pct: float = 0.55,
    plate_hex: Optional[str] = "#FFFFFF",
    plate_opacity: int = 180,
    # rayos
    rays_enabled: bool = True,
    rays_count: int = 12,
    rays_length_pct: float = 0.6,
    rays_thickness_px: int = 6,
    rays_color_hex: Optional[str] = "#FFD700",
    rays_opacity: int = 180,
    rays_spread_deg: float = 80.0,
    # packshot
    pack_scale_pct: float = 0.9,
    margin_right_pct: float = 0.06,
    margin_bottom_pct: float = 0.06,
    # sombra base
    shadow_scale_x: float = 0.9,
    shadow_scale_y: float = 0.08,
    shadow_offset_y_px: int = 6,
    shadow_opacity: int = 160,
    shadow_blur_px: int = 12,
) -> List[np.ndarray]:
    """
    Genera n creatividades usando SIEMPRE Vertex Imagen 3 para el fondo.
    - Packshot encima de la placa y rayos.
    - Sombra únicamente en la base.
    - Parámetros de placa, rayos, tamaño y posición del packshot configurables.
    """
    W, H = canvas_size
    outs: List[np.ndarray] = []

    for _ in range(max(1, int(n))):
        bg_img = _vertex_generate_background(
            W, H,
            prompt=bg_prompt,
            brand_hex=brand_hex,
            negative_prompt=bg_negative
        )
        arr = _compose_with_packshot(
            base_bytes=base_bytes,
            canvas_size=canvas_size,
            headline=headline,
            subheadline=subheadline,
            cta=cta,
            headline_hex=headline_hex,
            subheadline_hex=subheadline_hex,
            cta_hex=cta_hex,
            background_img=bg_img,
            quarter_radius_pct=quarter_radius_pct,
            plate_hex=plate_hex,
            plate_opacity=int(plate_opacity),
            rays_enabled=bool(rays_enabled),
            rays_count=int(rays_count),
            rays_length_pct=float(rays_length_pct),
            rays_thickness_px=int(rays_thickness_px),
            rays_color_hex=rays_color_hex,
            rays_opacity=int(rays_opacity),
            rays_spread_deg=float(rays_spread_deg),
            pack_scale_pct=float(pack_scale_pct),
            margin_right_pct=float(margin_right_pct),
            margin_bottom_pct=float(margin_bottom_pct),
            shadow_scale_x=float(shadow_scale_x),
            shadow_scale_y=float(shadow_scale_y),
            shadow_offset_y_px=int(shadow_offset_y_px),
            shadow_opacity=int(shadow_opacity),
            shadow_blur_px=int(shadow_blur_px),
        )
        outs.append(arr)

    return outs
