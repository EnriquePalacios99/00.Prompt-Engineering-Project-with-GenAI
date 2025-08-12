import os, re, json
from typing import Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION","global")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL","gemini-2.5-flash-lite")
FORCE_PUBLIC = os.getenv("FORCE_GEMINI_PUBLIC","").lower() in ("1","true","yes")

def _get_client_and_mode():
    from google import genai
    if FORCE_PUBLIC and GOOGLE_API_KEY:
        return genai.Client(api_key=GOOGLE_API_KEY), "public"
    if GCP_PROJECT:
        try:
            c = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
            _ = c.models.list()
            return c, "vertex"
        except Exception:
            if GOOGLE_API_KEY:
                return genai.Client(api_key=GOOGLE_API_KEY), "public"
            raise RuntimeError("Fallo Vertex y no hay GOOGLE_API_KEY para fallback.")
    if GOOGLE_API_KEY:
        return genai.Client(api_key=GOOGLE_API_KEY), "public"
    raise RuntimeError("Configura GCP_PROJECT o GOOGLE_API_KEY.")

def _build_contents(name:str, attrs:str, channel:str, images:Optional[List[bytes]]):
    from google.genai import types
    prompt = f"""Eres redactor de e-commerce en Perú.
    Producto: {name}
    Atributos: {attrs}
    Canal: {channel}
    Devuelve SOLO JSON válido con:
    {{"short":"","long":"","bullets":[],"hashtags":[]}}""".strip()
    parts = [types.Part.from_text(text=prompt)]
    if images:
        for b in images:
            parts.append(types.Part.from_bytes(data=b, mime_type="image/jpeg"))
    return [types.Content(role="user", parts=parts)]

def _extract_json(text:str)->dict:
    try: return json.loads(text)
    except: pass
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S|re.I)
    if m: return json.loads(m.group(1))
    m2 = re.search(r"(\{.*\})", text, flags=re.S)
    if m2: return json.loads(m2.group(1))
    raise ValueError("No se pudo extraer JSON")

def _normalize(out:dict)->dict:
    out.setdefault("short",""); out.setdefault("long","")
    out.setdefault("bullets",[]); out.setdefault("hashtags",[])
    if isinstance(out["bullets"], str):
        out["bullets"] = [x.strip() for x in re.split(r"\n|•|-|;", out["bullets"]) if x.strip()]
    if isinstance(out["hashtags"], str):
        out["hashtags"] = [t.strip() for t in re.split(r"[\s,]+", out["hashtags"]) if t.strip().startswith("#")]
    out["bullets"] = [str(x).strip() for x in out["bullets"] if str(x).strip()]
    out["hashtags"] = [str(x).strip() for x in out["hashtags"] if str(x).strip()]
    return out

def generate_product_description_gemini(name:str, attrs_text:str, channel:str, image_files:Optional[List[bytes]]=None,
                                       temperature:float=0.9, top_p:float=0.95, max_tokens:int=1024)->Dict:
    from google.genai import types
    client, _ = _get_client_and_mode()
    contents = _build_contents(name, attrs_text, channel, image_files)
    config = types.GenerateContentConfig(temperature=temperature, top_p=top_p, max_output_tokens=max_tokens)
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
    text = (resp.text or "").strip()
    try:
        parsed = _extract_json(text)
        out = _normalize(parsed); out["raw"] = text
        return out
    except Exception:
        return {"short":"","long":"","bullets":[],"hashtags":[],"raw":text}
