import os, re, json                              # Módulos estándar: entorno (os), expresiones regulares (re), y JSON (json)
from typing import Dict, List, Optional          # Tipos para anotaciones (mejor legibilidad/ayuda del IDE)
from dotenv import load_dotenv                   # Para cargar variables desde un archivo .env
load_dotenv()                                    # Carga las variables del archivo .env al entorno del proceso

GCP_PROJECT = os.getenv("GCP_PROJECT")           # ID del proyecto de Google Cloud (para usar Vertex AI)
GCP_LOCATION = os.getenv("GCP_LOCATION", "global")  # Región de Vertex AI (por defecto "global"; común: "us-central1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")     # Clave de la API pública de Gemini (google-genai)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")  # Modelo por defecto a usar para generación
FORCE_PUBLIC = os.getenv("FORCE_GEMINI_PUBLIC", "").lower() in ("1", "true", "yes")  # Si true/1/yes, fuerza API pública

# ============================================================================
# Función para obtener el cliente de Google Generative AI y el modo de conexión
# - Intenta primero usar Vertex AI si está configurado.
# - Si falla, usa la API pública con API key.
# - Lanza error si no hay credenciales válidas.
# ============================================================================
def _get_client_and_mode():
    from google import genai                     # Import local para evitar cargar si no se usa
    if FORCE_PUBLIC and GOOGLE_API_KEY:          # Si se fuerza API pública y existe API key...
        return genai.Client(api_key=GOOGLE_API_KEY), "public"  # ...usar cliente público

    if GCP_PROJECT:                               # Si hay proyecto configurado, intentar Vertex
        try:
            c = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)  # Cliente orientado a Vertex
            _ = c.models.list()                   # Llamada simple para validar acceso/permisos a Vertex
            return c, "vertex"                    # Si funciona, devolvemos cliente + modo "vertex"
        except Exception:                         # Si falla Vertex (falta API habilitada, permisos, etc.)
            if GOOGLE_API_KEY:                    # ...y tenemos API key pública
                return genai.Client(api_key=GOOGLE_API_KEY), "public"  # hacemos fallback a pública
            raise RuntimeError("Fallo Vertex y no hay GOOGLE_API_KEY para fallback.")  # sin fallback posible -> error

    if GOOGLE_API_KEY:                            # Si no hay proyecto pero sí API key, usar pública
        return genai.Client(api_key=GOOGLE_API_KEY), "public"

    raise RuntimeError("Configura GCP_PROJECT o GOOGLE_API_KEY.")  # Si no hay nada configurado, avisamos


# ============================================================================
# Función para construir el contenido del prompt para el modelo Gemini
# - Incluye formato, instrucciones y modificadores por canal.
# - Admite imágenes adicionales como entrada multimodal.
# ============================================================================
def _build_contents(name: str, attrs: str, channel: str, images: Optional[List[bytes]]):
    from google.genai import types               # Tipos del SDK para construir mensajes/partes

    # ========================= PROMPT MEJORADO (RATOS-D) =========================
    prompt = f"""
[ROL]
Eres un redactor senior de e-commerce en Perú, experto en conversión y SEO.

[AUDIENCIA]
Escribe en español peruano, claro y natural. Adapta el lenguaje al canal indicado.

[TAREA]
Redacta descripciones de producto a partir de:
- Nombre del producto
- Atributos declarados (NO inventar)
- Canal de publicación

[DATOS]
Producto: {name}
Atributos: {attrs}
Canal: {channel}

[REGLAS DE CONTENIDO]
- No inventes características que no estén en los atributos; si falta un dato, omítelo.
- Sin claims absolutos (“el mejor”, “#1”) ni beneficios de salud no sustentados.
- Evita MAYÚSCULAS sostenidas, emojis y signos excesivos.
- Moneda: usa “S/ 99.90” solo si aparece entre los atributos.
- SEO: incluye palabras clave naturales del producto (sin keyword stuffing).
- Tono: cercano pero profesional; gramática limpia; tuteo o neutro.

[FORMATO DE SALIDA — SOLO JSON VÁLIDO]
Devuelve EXCLUSIVAMENTE un JSON con esta estructura (sin backticks ni texto extra):
{{
  "short":   "≤160 caracteres; 1 frase con el principal valor percibido",
  "long":    "400–600 caracteres; 2–4 oraciones: qué es → beneficios → uso/mantenimiento",
  "bullets": ["4–6 viñetas; 6–14 palabras; concretas; sin punto final"],
  "hashtags":["5–8 hashtags en minúscula, sin acentos, con #; p. ej. #desayuno #cereales"]
}}

[MODIFICADORES POR CANAL]
- "IG": tono ligeramente más cercano; enfoca visualidad/ocasión de uso; evita cifras técnicas.
- "Ads": foco en “beneficio + CTA”; frases muy breves y contundentes.
- "Web": informativo y ordenado; añade especificaciones solo si están en atributos.
- "Marketplace": prioriza compatibilidades, tamaño, materiales, garantía si existen.

[CHECKLIST ANTES DE ENTREGAR]
- ¿El JSON parsea y usa comillas dobles?
- ¿short ≤160c? ¿long 400–600c?
- ¿4–6 bullets sin punto final?
- ¿5–8 hashtags; todos con #; sin espacios ni acentos?
- ¿Nada fuera de lo declarado en atributos?

Devuelve solo el JSON.
""".strip()
    # ============================================================================

    # (Opcional) Few-shot para mayor estabilidad: actívalo con PROMPT_FEWSHOT_EXAMPLE=1 en .env
    if os.getenv("PROMPT_FEWSHOT_EXAMPLE", "").lower() in ("1", "true", "yes"):
        few_shot = """
[EJEMPLO — NO COPIAR LITERALMENTE, SOLO REFERENCIA DE ESTILO]
Entrada:
- Producto: Cereal integral de avena y quinua 300 g
- Atributos: fibra 6 g por porción; sin azúcar añadida; bolsa 300 g; marca Ángel; ideal desayuno
- Canal: Web

Salida (JSON):
{
  "short":"Energía natural para tus mañanas con fibra y sabor auténtico.",
  "long":"Cereal integral de avena y quinua para un desayuno nutritivo y sabroso. Su aporte de fibra ayuda a mantener la saciedad por más tiempo y acompaña una rutina equilibrada. Disfrútalo con leche o yogurt y combínalo con fruta fresca. Presentación de 300 g práctica para tu despensa.",
  "bullets":[
    "avena y quinua integrales",
    "fuente de fibra por porcion",
    "sin azucar anadida",
    "formato 300 g facil de guardar",
    "ideal para desayuno o snack"
  ],
  "hashtags":["#desayuno","#cereal","#fibra","#avena","#quinua","#vidasana"]
}
""".strip()
        prompt = prompt + "\n\n" + few_shot

    parts = [types.Part.from_text(text=prompt)]  # Creamos la parte de texto (instrucciones/prompt)
    if images:                                   # Si se pasaron imágenes (bytes)...
        for b in images:
            parts.append(types.Part.from_bytes(data=b, mime_type="image/jpeg"))  # ...adjuntarlas como partes multimodales
    return [types.Content(role="user", parts=parts)]  # Construimos el "Content" del usuario con sus parts


# ============================================================================
# Función para extraer un JSON válido desde el texto de salida del modelo
# - Intenta parseo directo.
# - Busca bloques ```json ...``` si el directo falla.
# - Si todo falla, lanza error.
# ============================================================================
def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)                  # 1) Intentar parsear directamente como JSON
    except:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)  # 2) Buscar bloque ```json ... ```
    if m:
        return json.loads(m.group(1))            #    Si hay fence code con JSON, parsearlo
    m2 = re.search(r"(\{.*\})", text, flags=re.S) # 3) Buscar el primer bloque {...} que parezca JSON
    if m2:
        return json.loads(m2.group(1))           #    Si lo encuentra, parsearlo
    raise ValueError("No se pudo extraer JSON")  # 4) Si nada funcionó, avisar con error


# ============================================================================
# Función para normalizar la estructura JSON resultante
# - Asegura que las claves mínimas existan.
# - Convierte strings en listas si es necesario.
# - Limpia espacios y elementos vacíos.
# ============================================================================
def _normalize(out: dict) -> dict:
    out.setdefault("short", ""); out.setdefault("long", "")
    out.setdefault("bullets", []); out.setdefault("hashtags", [])

    if isinstance(out["bullets"], str):
        out["bullets"] = [x.strip() for x in re.split(r"\n|•|-|;", out["bullets"]) if x.strip()]
    if isinstance(out["hashtags"], str):
        out["hashtags"] = [t.strip() for t in re.split(r"[\s,]+", out["hashtags"]) if t.strip().startswith("#")]

    out["bullets"] = [str(x).strip() for x in out["bullets"] if str(x).strip()]
    out["hashtags"] = [str(x).strip() for x in out["hashtags"] if str(x).strip()]
    return out


# ============================================================================
# Función principal para generar descripciones de producto usando Gemini
# - Recibe datos del producto, canal y opcionalmente imágenes.
# - Construye el prompt y llama al modelo configurado.
# - Devuelve la respuesta parseada y normalizada.
# ============================================================================
def generate_product_description_gemini(name: str, attrs_text: str, channel: str,
                                        image_files: Optional[List[bytes]] = None,
                                        temperature: float = 0.9, top_p: float = 0.95,
                                        max_tokens: int = 1024) -> Dict:
    from google.genai import types
    client, _ = _get_client_and_mode()                          # Obtener cliente y modo
    contents = _build_contents(name, attrs_text, channel, image_files)  # Construir prompt multimodal
    config = types.GenerateContentConfig(temperature=temperature, top_p=top_p, max_output_tokens=max_tokens)
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
    text = (resp.text or "").strip()

    try:
        parsed = _extract_json(text)                            # Intentar extraer JSON
        out = _normalize(parsed)
        out["raw"] = text                                       # Guardar texto original para depuración
        return out
    except Exception:
        return {"short": "", "long": "", "bullets": [], "hashtags": [], "raw": text}  # Fallback mínimo
