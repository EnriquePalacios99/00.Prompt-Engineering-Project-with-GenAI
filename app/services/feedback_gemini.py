import os, json, re
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()
GCP_PROJECT=os.getenv("GCP_PROJECT"); GCP_LOCATION=os.getenv("GCP_LOCATION","global")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"); GEMINI_MODEL=os.getenv("GEMINI_MODEL","gemini-2.5-flash-lite")

def _client():
    from google import genai
    if GCP_PROJECT:
        try:
            c=genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION); _=c.models.list(); return c
        except Exception:
            if GOOGLE_API_KEY: return genai.Client(api_key=GOOGLE_API_KEY)
            raise
    if GOOGLE_API_KEY: return genai.Client(api_key=GOOGLE_API_KEY)
    raise RuntimeError("Configura GCP_PROJECT o GOOGLE_API_KEY.")

def summarize_reviews_gemini(reviews: List[str]) -> str:
    from google.genai import types
    c=_client()
    prompt=("Eres analista de CX. Resume en 3-5 bullets y 1 recomendación. Español conciso.\n"
            "REVIEWS_JSON:\n"+json.dumps(reviews[:300], ensure_ascii=False))
    resp=c.models.generate_content(model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(prompt)])],
            config=types.GenerateContentConfig(max_output_tokens=512, temperature=0.4))
    return (resp.text or "").strip()

def score_sentiment_gemini(reviews: List[str]) -> List[Dict]:
    from google.genai import types
    c=_client()
    sys=('Clasifica cada review en positivo/negativo/neutral. Devuelve SOLO JSON: '
         '[{"review":"","sentiment":"","rationale":""}]')
    prompt=sys+"\nREVIEWS_JSON:\n"+json.dumps(reviews[:200], ensure_ascii=False)
    resp=c.models.generate_content(model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(prompt)])],
            config=types.GenerateContentConfig(max_output_tokens=2048, temperature=0.2))
    text=(resp.text or "").strip()
    try: return json.loads(text)
    except Exception:
        m=re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, flags=re.I)
        if m: return json.loads(m.group(1))
        m2=re.search(r"(\[[\s\S]*\])", text)
        if m2: return json.loads(m2.group(1))
        return [{"review": r[:160], "sentiment":"neutral", "rationale":""} for r in reviews[:50]]
