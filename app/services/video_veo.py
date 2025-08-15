# app/services/video_veo.py
import os
import time
import imghdr
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Si quieres que Veo escriba el resultado en GCS (cuando el SDK no devuelve bytes inline):
# Ejemplo: gs://mi-bucket/salidas/veo  (el bucket debe existir y el SA tener permisos)
OUTPUT_GCS_URI = os.getenv("OUTPUT_GCS_URI", "").strip()

# Modelos recomendados (ajústalos según tu región/disponibilidad)
DEFAULT_TEXT2VIDEO_MODEL = os.getenv("VEO_TEXT_MODEL", "veo-3.0-fast-generate-001")
DEFAULT_IMG2VIDEO_MODEL  = os.getenv("VEO_IMAGE_MODEL", "veo-2.0-generate-001")


def _client():
    """Crea el cliente google-genai. Prioriza Vertex (project+location),
    si falla, intenta con Gemini API (API key)."""
    from google import genai
    if GCP_PROJECT:
        try:
            c = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
            _ = c.models.list()
            return c
        except Exception:
            if GOOGLE_API_KEY:
                return genai.Client(api_key=GOOGLE_API_KEY)
            raise
    if GOOGLE_API_KEY:
        return genai.Client(api_key=GOOGLE_API_KEY)
    raise RuntimeError("Configura GCP_PROJECT o GOOGLE_API_KEY para usar Veo.")


def _guess_mime_from_bytes(b: bytes) -> str:
    """Inferir MIME del packshot subido."""
    kind = imghdr.what(None, h=b)
    if kind == "png":
        return "image/png"
    if kind in ("jpg", "jpeg"):
        return "image/jpeg"
    return "image/jpeg"


def _build_marketing_prompt(base_prompt: str, brand: str = "", product: str = "", style: str = "") -> str:
    """Prompt RATOS-D condensado para videos promocionales."""
    blocks = []
    blocks.append("[ROL] Director creativo de video comercial.")
    blocks.append("[AUDIENCIA] Público general en redes sociales (Perú).")
    blocks.append("[TAREA] Generar un spot corto mostrando el producto con movimientos suaves y enfoque en beneficios.")
    if product or brand:
        blocks.append(f"[OBJETIVO] Resaltar {product or 'el producto'} de {brand or 'la marca'} con encuadres agradables.")
    if style:
        blocks.append(f"[ESTILO] {style}.")
    if base_prompt:
        blocks.append(f"[GUIA VISUAL] {base_prompt}")
    blocks.append("[DO] Movimiento de cámara leve, iluminación limpia, enfoque en el packshot, ritmo natural.")
    blocks.append("[DONTS] Evitar texto en pantalla y logos inventados. Nada de rostros humanos.")
    return "\n".join(blocks)


def _build_videos_config_safe(
    *,
    number_of_videos: Optional[int] = None,
    duration_seconds: Optional[int] = None,
    fps: Optional[int] = None,
    enhance_prompt: Optional[bool] = None,
    negative_prompt: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    generate_audio: Optional[bool] = None,
    seed: Optional[int] = None,
):
    """Construye GenerateVideosConfig probando claves conocidas.
    Si el SDK rechaza alguna (pydantic extra_forbidden), reintenta con un subconjunto."""
    from google.genai import types

    # Intento 1: completo (sin video_resolution)
    try:
        return types.GenerateVideosConfig(
            **{k: v for k, v in {
                "number_of_videos": number_of_videos,
                "duration_seconds": duration_seconds,
                "fps": fps,
                "enhance_prompt": enhance_prompt,
                "negative_prompt": negative_prompt,
                "aspect_ratio": aspect_ratio,    # algunos builds lo aceptan
                "generate_audio": generate_audio, # algunos builds lo aceptan
                "seed": seed,
            }.items() if v is not None}
        )
    except Exception:
        pass

    # Intento 2: sin aspect_ratio / generate_audio
    try:
        return types.GenerateVideosConfig(
            **{k: v for k, v in {
                "number_of_videos": number_of_videos,
                "duration_seconds": duration_seconds,
                "fps": fps,
                "enhance_prompt": enhance_prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
            }.items() if v is not None}
        )
    except Exception:
        pass

    # Intento 3: mínimo
    return types.GenerateVideosConfig(
        **{k: v for k, v in {
            "negative_prompt": negative_prompt,
        }.items() if v is not None}
    )


def generate_promo_videos(
    *,
    prompt: str,
    negative_prompt: str = "",
    product_image_bytes: Optional[bytes] = None,
    model: Optional[str] = None,
    aspect_ratio: str = "16:9",           # "16:9", "9:16", "1:1" (si no lo soporta tu SDK, se ignora)
    duration_seconds: int = 8,            # Veo 3 produce hasta ~8s a 720p
    resolution: str = "720p",             # Referencial (el SDK Python puede ignorarlo; Veo 3 entrega 720p por defecto)
    number_of_videos: int = 1,
    generate_audio: bool = False,         # Si tu despliegue lo soporta
    brand: str = "",
    product_name: str = "",
    style_hint: str = "",
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Genera N videos; si product_image_bytes → imagen→video; si no → texto→video.
    Devuelve: [{"video_bytes": bytes|None, "mime_type":"video/mp4", "gcs_uri": str|None, ...}, ...]"""
    from google.genai import types

    client = _client()
    is_img2video = product_image_bytes is not None
    model_id = model or (DEFAULT_IMG2VIDEO_MODEL if is_img2video else DEFAULT_TEXT2VIDEO_MODEL)

    # Prompt mejorado
    full_prompt = _build_marketing_prompt(
        base_prompt=prompt, brand=brand, product=product_name, style=style_hint
    )

    # Config segura (sin video_resolution)
    cfg = _build_videos_config_safe(
        number_of_videos=max(1, int(number_of_videos)),
        duration_seconds=int(duration_seconds),
        fps=24,
        enhance_prompt=True,
        negative_prompt=negative_prompt or None,
        aspect_ratio=aspect_ratio or None,
        generate_audio=bool(generate_audio) if generate_audio else None,
        seed=seed,
    )

    # Armar kwargs para generate_videos
    gen_kwargs: Dict[str, Any] = dict(
        model=model_id,
        prompt=full_prompt,
        config=cfg,
    )
    if is_img2video:
        gen_kwargs["image"] = types.Image(
            image_bytes=product_image_bytes,
            mime_type=_guess_mime_from_bytes(product_image_bytes)
        )
    if OUTPUT_GCS_URI:
        # Algunos despliegues requieren salida a GCS; el SDK puede escribir ahí.
        gen_kwargs["output_gcs_uri"] = OUTPUT_GCS_URI.rstrip("/")

    # 1) Lanzar operación
    op = client.models.generate_videos(**gen_kwargs)

    # 2) Polling hasta terminar
    while not op.done:
        time.sleep(15)
        op = client.operations.get(op)

    if not getattr(op, "result", None):
        raise RuntimeError(f"Veo no devolvió resultado. Detalle: {getattr(op, 'error', 'sin error adjunto')}")

    # 3) Extraer videos
    results: List[Dict[str, Any]] = []
    for item in getattr(op.result, "generated_videos", []):
        video = getattr(item, "video", None)
        gcs_uri = getattr(getattr(video, "uri", None), "uri", None) or getattr(video, "uri", None)

        video_bytes = None
        for attr in ("bytes", "video_bytes", "_video_bytes"):
            if hasattr(video, attr) and getattr(video, attr):
                video_bytes = getattr(video, attr)
                break
        if video_bytes is None and hasattr(video, "to_bytes"):
            try:
                video_bytes = video.to_bytes()
            except Exception:
                pass

        results.append({
            "video_bytes": video_bytes,
            "mime_type": getattr(video, "mime_type", "video/mp4"),
            "gcs_uri": gcs_uri if OUTPUT_GCS_URI else (gcs_uri or None),
            "model": model_id,
            "duration": duration_seconds,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,  # informativo
        })

    if not results:
        raise RuntimeError("La operación terminó sin videos generados.")

    return results
