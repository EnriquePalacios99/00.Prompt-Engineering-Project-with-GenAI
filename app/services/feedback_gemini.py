# services/feedback_gemini.py
import os, json, re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "global")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


# -----------------------------
# Cliente google-genai (Vertex o pública)
# -----------------------------
def _client():
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
    raise RuntimeError("Configura GCP_PROJECT o GOOGLE_API_KEY.")


# ---------- Helpers de compatibilidad para construir parts/contents ----------
def _make_text_part(text: str):
    """Devuelve un 'part' de texto compatible con varias versiones del SDK."""
    from google.genai import types
    try:
        return types.Part.from_text(text=text)
    except Exception:
        pass
    try:
        return types.Part(text=text)
    except Exception:
        pass
    return text  # fallback: el cliente acepta strings


def _make_image_part(b: bytes, mime: str = "image/jpeg"):
    """Devuelve un 'part' de imagen compatible con varias versiones del SDK."""
    from google.genai import types
    try:
        return types.Part.from_bytes(data=b, mime_type=mime)
    except Exception:
        pass
    try:
        return types.Part(inline_data={"mime_type": mime, "data": b})
    except Exception:
        pass
    raise RuntimeError("No se pudo construir Part de imagen con la versión instalada de google-genai")


def _build_contents_robusto(prompt: str, images: Optional[List[bytes]] = None):
    """
    Si no hay imágenes, devolvemos el string directamente (máx. compatibilidad).
    Si hay imágenes, construimos un Content con parts compatibles.
    """
    if not images:
        return prompt
    from google.genai import types
    parts = [_make_text_part(prompt)]
    for b in images:
        parts.append(_make_image_part(b, mime="image/jpeg"))
    return [types.Content(role="user", parts=parts)]


# -----------------------------
# Utilidades de extracción JSON
# -----------------------------
def _extract_json_obj(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.I)
    if m:
        return json.loads(m.group(1))
    m2 = re.search(r"(\{[\s\S]*\})", text)
    if m2:
        return json.loads(m2.group(1))
    raise ValueError("No se pudo extraer JSON (obj)")


def _extract_json_arr(text: str) -> List[Dict[str, Any]]:
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, flags=re.I)
    if m:
        return json.loads(m.group(1))
    m2 = re.search(r"(\[[\s\S]*\])", text)
    if m2:
        return json.loads(m2.group(1))
    raise ValueError("No se pudo extraer JSON (array)")


def _clip(s: str, n: int = 160) -> str:
    s = (s or "").strip().replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s


# -----------------------------
# Resumen de reviews (RATOS-D) → JSON
# -----------------------------
def summarize_reviews_gemini(
    reviews: List[str],
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_output_tokens: int = 768,
    max_reviews: int = 300
) -> Dict[str, Any]:
    """
    Devuelve un dict JSON:
    {
      "bullets": ["...", "...", "..."],  # 3–5
      "recommendation": "1 párrafo con acción prioritaria",
      "sentiment_ratio": {"positivo":0.00,"neutral":0.00,"negativo":0.00},
      "action_plan": ["Paso; responsable; plazo", ...],  # 3–5
      "customer_reply": "Párrafo 3–6 oraciones, empático y profesional",
      "sample_size": int,
      "raw": "texto de salida del modelo (debug)"
    }
    """
    from google.genai import types
    c = _client()

    subset = [str(x) for x in reviews[:max_reviews]]

    prompt = f"""
[ROL] Analista senior de Customer Experience en Perú.
[AUDIENCIA] Equipo de producto/marketing y atención al cliente.
[TAREA]
1) Resume comentarios en 3–5 bullets.
2) Da 1 recomendación prioritaria (parrafo corto).
3) Propón un plan de acción en 3–5 pasos (con responsable y plazo sugerido).
4) Redacta una respuesta pública al cliente (3–6 oraciones, tono empático/profesional, sin admitir culpa legal).
[DATOS] REVIEWS_JSON (UTF-8): {json.dumps(subset, ensure_ascii=False)}
[REGLAS]
- Español claro, conciso, sin jerga técnica.
- No inventes; usa solo patrones repetidos.
- En la respuesta pública: agradece, reconoce la experiencia, ofrece canal de contacto y pide datos (orden, contacto) si corresponde.
[FORMATO DE SALIDA — SOLO JSON]
Devuelve únicamente:
{{
  "bullets": ["3 a 5 bullets; 8–18 palabras; sin punto final"],
  "recommendation": "1 párrafo con la acción prioritaria para mejorar CX",
  "sentiment_ratio": {{"positivo":0.0,"neutral":0.0,"negativo":0.0}},
  "action_plan": ["Paso; Responsable; Plazo (ej. 2 semanas)", "…"],
  "customer_reply": "Respuesta pública breve (3–6 oraciones, tono empático)",
  "sample_size": {len(subset)}
}}
[CHECKLIST]
- ¿JSON válido? ¿3–5 bullets? ¿3–5 pasos en plan? ¿respuesta 3–6 oraciones? ¿sin texto extra?
""".strip()

    contents = _build_contents_robusto(prompt, images=None)
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )
    resp = c.models.generate_content(model=GEMINI_MODEL, contents=contents, config=cfg)
    text = (resp.text or "").strip()

    # Parse + normalización
    try:
        data = _extract_json_obj(text)
    except Exception:
        # Fallback mínimo
        bullets = []
        for s in subset[:5]:
            s = re.sub(r"\s+", " ", s)
            if s:
                bullets.append(_clip(s, 90))
        return {
            "bullets": bullets[:5] or ["Sin datos"],
            "recommendation": "Revisar comentarios negativos y priorizar mejoras repetidas.",
            "sentiment_ratio": {"positivo": 0.33, "neutral": 0.34, "negativo": 0.33},
            "action_plan": ["Revisar tickets abiertos; CX; 2 semanas"],
            "customer_reply": "¡Gracias por tu comentario! Queremos ayudarte. Escríbenos por DM con tu número de pedido para revisar tu caso y darte una solución.",
            "sample_size": len(subset),
            "raw": text
        }

    # Bullets
    bullets = data.get("bullets") or []
    if isinstance(bullets, str):
        bullets = [x.strip(" •-") for x in re.split(r"[\n;•-]+", bullets) if x.strip()]
    bullets = [str(b).strip().rstrip(".") for b in bullets if str(b).strip()]
    if len(bullets) > 5:
        bullets = bullets[:5]

    # Recomendación
    reco = str(data.get("recommendation", "")).strip()
    if not reco:
        reco = "Prioriza una intervención concreta sobre el punto de mayor fricción repetido."

    # Ratios
    sr = data.get("sentiment_ratio") or {}
    def _f(k):
        try:
            return float(sr.get(k, 0))
        except Exception:
            return 0.0
    srn = {"positivo": _f("positivo"), "neutral": _f("neutral"), "negativo": _f("negativo")}
    ssum = sum(srn.values())
    if ssum > 0:
        srn = {k: v / ssum for k, v in srn.items()}
    else:
        srn = {"positivo": 0.33, "neutral": 0.34, "negativo": 0.33}

    # Plan de acción
    plan = data.get("action_plan") or []
    if isinstance(plan, str):
        plan = [x.strip(" •-") for x in re.split(r"[\n;•-]+", plan) if x.strip()]
    plan = [str(p).strip().rstrip(".") for p in plan if str(p).strip()]
    plan = plan[:5] if len(plan) > 5 else plan
    if not plan:
        plan = ["Revisar tickets abiertos; CX; 2 semanas"]

    # Respuesta pública
    reply = str(data.get("customer_reply", "")).strip()
    if not reply:
        reply = "¡Gracias por tu comentario! Queremos ayudarte. Escríbenos por DM con tu número de pedido para revisar tu caso y darte una solución."

    out = {
        "bullets": bullets or ["Sin hallazgos destacables."],
        "recommendation": reco,
        "sentiment_ratio": srn,
        "action_plan": plan,
        "customer_reply": reply,
        "sample_size": int(data.get("sample_size") or len(subset)),
        "raw": text
    }
    return out


# -----------------------------
# Scoring de sentimiento (RATOS-D) → lista de dicts
# -----------------------------
_LABELS = {"positivo", "negativo", "neutral"}
_SYNONYMS = {
    "pos": "positivo", "positive": "positivo", "+": "positivo",
    "neg": "negativo", "negative": "negativo", "-": "negativo",
    "neu": "neutral", "0": "neutral"
}

def _normalize_label(x: str) -> str:
    xl = (x or "").strip().lower()
    if xl in _LABELS:
        return xl
    return _SYNONYMS.get(xl, "neutral")


def score_sentiment_gemini(
    reviews: List[str],
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_output_tokens: int = 2048,
    max_reviews: int = 200
) -> List[Dict]:
    """
    Devuelve: [{"review":"(≤160c)","sentiment":"positivo|neutral|negativo","rationale":"..."}]
    """
    from google.genai import types
    c = _client()

    subset = [str(x) for x in reviews[:max_reviews]]
    sys = """
[ROL] Analista de sentimiento.
[TAREA] Clasifica cada review en 'positivo', 'negativo' o 'neutral' y explica brevemente por qué.
[REGLAS]
- Español conciso (1–2 frases de justificación).
- No inventes hechos que no estén en el texto.
[FORMATO — SOLO JSON]
Devuelve SOLO un array JSON:
[
  {"review":"texto original (recorte ≤160 caracteres)","sentiment":"positivo|neutral|negativo","rationale":"motivo breve"}
]
[CHECKLIST]
- ¿JSON válido? ¿Etiquetas SOLO entre positivo/neutral/negativo? ¿sin texto extra?
""".strip()

    prompt = sys + "\nREVIEWS_JSON:\n" + json.dumps(subset, ensure_ascii=False)
    contents = _build_contents_robusto(prompt, images=None)
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )
    resp = c.models.generate_content(model=GEMINI_MODEL, contents=contents, config=cfg)
    text = (resp.text or "").strip()

    try:
        rows = _extract_json_arr(text)
    except Exception:
        return [{"review": _clip(r), "sentiment": "neutral", "rationale": ""} for r in subset[:50]]

    clean: List[Dict[str, Any]] = []
    for r in rows:
        review = _clip(str(r.get("review") or ""))
        label = _normalize_label(str(r.get("sentiment") or "neutral"))
        rationale = _clip(str(r.get("rationale") or ""), 240)
        if not review:
            continue
        clean.append({"review": review, "sentiment": label, "rationale": rationale})

    if not clean:
        clean = [{"review": _clip(r), "sentiment": "neutral", "rationale": ""} for r in subset[:50]]
    return clean


# -----------------------------
# Respuesta a un comentario individual
# -----------------------------
def generate_customer_reply_gemini(
    comment: str,
    brand_name: Optional[str] = None,
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_output_tokens: int = 512
) -> Dict[str, str]:
    """
    Devuelve: {"reply":"...", "notes":"(opcional)"}
    """
    from google.genai import types
    c = _client()

    prompt = f"""
[ROL] Agente senior de atención al cliente.
[TAREA] Redacta una respuesta pública breve (3–6 oraciones) al comentario del cliente.
[DATOS] COMENTARIO: {comment}
[REGLAS]
- Español peruano, tono empático y profesional.
- Agradecer el feedback; reconocer la experiencia; ofrecer ayuda.
- Si es un problema, ofrece canal directo y solicita datos (orden, contacto).
- No admitir culpa legal; no prometer algo que no exista.
- Mantener 3–6 oraciones; evitar emojis y mayúsculas sostenidas.
[CONTEXTO MARCA]
- Marca: {brand_name or "n/a"}

[FORMATO — SOLO JSON]
{{
  "reply": "texto de respuesta pública (3–6 oraciones)"
}}
""".strip()

    contents = _build_contents_robusto(prompt, images=None)
    cfg = types.GenerateContentConfig(
        temperature=temperature, top_p=top_p, max_output_tokens=max_output_tokens
    )
    resp = c.models.generate_content(model=GEMINI_MODEL, contents=contents, config=cfg)
    text = (resp.text or "").strip()

    try:
        data = _extract_json_obj(text)
        reply = str(data.get("reply", "")).strip()
        if not reply:
            raise ValueError("sin campo reply")
        return {"reply": reply}
    except Exception:
        # fallback simple
        return {
            "reply": "¡Gracias por escribirnos! Queremos ayudarte con tu caso. Por favor envíanos un mensaje directo con tu número de pedido y datos de contacto para revisar lo ocurrido y darte una solución."
        }
