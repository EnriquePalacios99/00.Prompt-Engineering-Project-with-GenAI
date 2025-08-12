# scaffold_genai_project.py
import os, textwrap, json

ROOT = "."
PKG = "app"

FILES = {
    # --- app skeleton ---
    f"{PKG}/__init__.py": "",
    f"{PKG}/app.py": textwrap.dedent("""\
        import streamlit as st
        st.set_page_config(page_title="GenAI MVP", layout="wide")
        st.title("GenAI MVP – Descripciones, Imágenes y Feedback")
        st.markdown(
            "- **01_Descripciones**: genera copys listos para e-commerce.\\n"
            "- **02_Imagenes**: produce creatividades promocionales a partir de un packshot.\\n"
            "- **03_Feedback**: resume comentarios y clasifica sentimiento."
        )
        st.info("Configura tus credenciales en `.env` en la raíz del proyecto.")
    """),

    # --- pages ---
    f"{PKG}/pages/__init__.py": "",
    f"{PKG}/pages/01_Descripciones.py": textwrap.dedent("""\
        import streamlit as st
        from services.llm_gemini import generate_product_description_gemini

        st.title("Generación de descripciones (Gemini)")
        name = st.text_input("Nombre del producto", "Cereales Ángel")
        attrs = st.text_area("Atributos (texto/JSON breve)", "fortificado, crujiente, familiar")
        channel = st.selectbox("Canal", ["ecommerce", "marketplace", "redes"])
        images = st.file_uploader("Imágenes del producto (opcional)", type=["jpg","jpeg","png"], accept_multiple_files=True)

        if st.button("Generar", type="primary"):
            img_bytes = [f.read() for f in (images or [])]
            out = generate_product_description_gemini(name, attrs, channel, img_bytes)
            st.subheader("Descripción corta"); st.write(out.get("short",""))
            st.subheader("Descripción larga (SEO)"); st.write(out.get("long",""))
            st.subheader("Bullets"); st.write(out.get("bullets", []))
            st.subheader("Hashtags"); st.write(out.get("hashtags", []))
            with st.expander("Salida completa (raw)"):
                st.code(out.get("raw",""), language="json")
    """),
    f"{PKG}/pages/02_Imagenes.py": textwrap.dedent("""\
        import streamlit as st
        from services.images_gemini import generate_promos_with_gemini_background

        st.title("Imágenes promocionales")
        base = st.file_uploader("Sube packshot (PNG/JPG)", type=["png","jpg","jpeg"])
        formato = st.selectbox("Formato", ["1080x1350 (IG Feed)", "1200x628 (Ads)"])
        headline = st.text_input("Titular", "Nuevo Cereales Ángel")
        subheadline = st.text_input("Subtítulo", "Más sabor para tus mañanas")
        cta = st.text_input("CTA", "Compra ahora")
        n = st.number_input("N° de imágenes", 1, 4, 1)
        brand_hex = st.color_picker("Color principal", "#E30613")

        if st.button("Generar", type="primary"):
            if not base:
                st.warning("Sube un packshot para continuar."); st.stop()
            size = (1080,1350) if "1080x1350" in formato else (1200,628)
            imgs = generate_promos_with_gemini_background(
                base.read(), headline, subheadline, cta, int(n), size, brand_hex
            )
            for i, arr in enumerate(imgs, 1):
                st.image(arr, caption=f"Creatividad {i}")
    """),
    f"{PKG}/pages/03_Feedback.py": textwrap.dedent("""\
        import streamlit as st, pandas as pd
        # Fallback local
        from services.feedback import summarize_reviews as sum_local, score_sentiment as sent_local
        # Gemini (si hay credenciales)
        try:
            from services.feedback_gemini import summarize_reviews_gemini as sum_gem, score_sentiment_gemini as sent_gem
            USE_GEMINI = True
        except Exception:
            USE_GEMINI = False

        st.title("Feedback de clientes")
        up = st.file_uploader("CSV con columna de texto (review/comentario/...)", type=["csv"])

        def _read(file):
            df = pd.read_csv(file, sep=None, engine="python")
            if df.shape[1]==1 and ";" in df.columns[0]:
                file.seek(0); df = pd.read_csv(file, sep=";")
            df.columns = [c.strip().lower() for c in df.columns]
            return df

        if up:
            df = _read(up)
            st.dataframe(df.head())
            cols = df.columns.tolist()
            guess = next((c for c in cols if c in ["review","comentario","texto","opinion","comment"]), cols[0])
            text_col = st.selectbox("Selecciona la columna de texto", cols, index=cols.index(guess))
            reviews = df[text_col].dropna().astype(str).tolist()

            c1,c2 = st.columns(2)
            with c1:
                if st.button("Resumen"):
                    try:
                        res = sum_gem(reviews) if USE_GEMINI else sum_local(reviews)
                    except Exception:
                        res = sum_local(reviews)
                    st.subheader("Resumen"); st.write(res)
            with c2:
                if st.button("Sentimiento"):
                    try:
                        rows = sent_gem(reviews) if USE_GEMINI else sent_local(reviews)
                    except Exception:
                        rows = sent_local(reviews)
                    st.subheader("Análisis de sentimiento"); st.dataframe(pd.DataFrame(rows))
    """),

    # --- services ---
    f"{PKG}/services/__init__.py": "",
    f"{PKG}/services/llm_gemini.py": textwrap.dedent("""\
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
            prompt = f\"\"\"Eres redactor de e-commerce en Perú.
            Producto: {name}
            Atributos: {attrs}
            Canal: {channel}
            Devuelve SOLO JSON válido con:
            {{"short":"","long":"","bullets":[],"hashtags":[]}}\"\"\".strip()
            parts = [types.Part.from_text(text=prompt)]
            if images:
                for b in images:
                    parts.append(types.Part.from_bytes(data=b, mime_type="image/jpeg"))
            return [types.Content(role="user", parts=parts)]

        def _extract_json(text:str)->dict:
            try: return json.loads(text)
            except: pass
            import re
            m = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", text, flags=re.S|re.I)
            if m: return json.loads(m.group(1))
            m2 = re.search(r"(\\{.*\\})", text, flags=re.S)
            if m2: return json.loads(m2.group(1))
            raise ValueError("No se pudo extraer JSON")

        def _normalize(out:dict)->dict:
            out.setdefault("short",""); out.setdefault("long","")
            out.setdefault("bullets",[]); out.setdefault("hashtags",[])
            if isinstance(out["bullets"], str):
                out["bullets"] = [x.strip() for x in re.split(r"\\n|•|-|;", out["bullets"]) if x.strip()]
            if isinstance(out["hashtags"], str):
                out["hashtags"] = [t.strip() for t in re.split(r"[\\s,]+", out["hashtags"]) if t.strip().startswith("#")]
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
    """),
    f"{PKG}/services/images_gemini.py": textwrap.dedent("""\
        from typing import List, Tuple
        import numpy as np
        from io import BytesIO
        from PIL import Image, ImageDraw, ImageFont, ImageFilter

        def _hex_to_rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
        def _font(sz):
            try: from PIL import ImageFont; return ImageFont.truetype("DejaVuSans-Bold.ttf", sz)
            except: return ImageFont.load_default()

        def _bg(size:Tuple[int,int], color:Tuple[int,int,int])->Image.Image:
            return Image.new("RGB", size, color)

        def _fit_shadow(img:Image.Image, max_w:int, max_h:int)->Image.Image:
            img = img.convert("RGBA"); img.thumbnail((max_w,max_h), Image.LANCZOS)
            shadow = Image.new("RGBA",(img.width+30,img.height+30),(0,0,0,0))
            s = Image.new("RGBA",(img.width,img.height),(0,0,0,120))
            shadow.paste(s,(15,15),s); shadow = shadow.filter(ImageFilter.GaussianBlur(10))
            can = Image.new("RGBA",shadow.size,(0,0,0,0)); can.alpha_composite(shadow,(0,0)); can.alpha_composite(img,(0,0))
            return can

        def generate_promos_with_gemini_background(base_bytes:bytes, headline:str, subheadline:str, cta:str, n:int,
                                                   canvas_size:Tuple[int,int], brand_hex:str)->List[np.ndarray]:
            brand = _hex_to_rgb(brand_hex)
            prod = Image.open(BytesIO(base_bytes)).convert("RGBA")
            outs=[]
            for _ in range(n):
                bg = _bg(canvas_size, (245,245,245)).convert("RGBA")
                W,H = bg.size
                pack = _fit_shadow(prod, int(W*0.6), int(H*0.55))
                bg.alpha_composite(pack, (int(W*0.55), int(H*0.18)))
                d = ImageDraw.Draw(bg)
                fH, fS, fC = _font(int(H*0.06)), _font(int(H*0.035)), _font(int(H*0.035))
                x0, y0, text_w = int(W*0.07), int(H*0.18), int(W*0.42)

                def draw_wrap(text, box, font, fill=(20,20,20)):
                    x0,y0,x1,y1 = box; max_w = x1-x0; words = text.split(); cur=""; lines=[]
                    for w in words:
                        t=(cur+" "+w).strip()
                        if d.textlength(t, font=font) <= max_w: cur=t
                        else: lines.append(cur); cur=w
                    if cur: lines.append(cur)
                    y=y0; lh=(font.size+6 if hasattr(font,"size") else 22)
                    for ln in lines:
                        d.text((x0,y), ln, font=font, fill=fill); y+=lh

                draw_wrap(headline,(x0,y0,x0+text_w,y0+int(H*0.18)), fH, (20,20,20))
                y0+=int(H*0.16)
                draw_wrap(subheadline,(x0,y0,x0+text_w,y0+int(H*0.16)), fS, (60,60,60))
                y0+=int(H*0.14)

                # CTA
                btn_w, btn_h = int(text_w*0.75), int(H*0.08); btn_x, btn_y = x0, y0
                d.rounded_rectangle([btn_x,btn_y,btn_x+btn_w,btn_y+btn_h], radius=int(btn_h/2), fill=(255,255,255,230))
                tw = d.textlength(cta, font=fC)
                d.text((btn_x+(btn_w-tw)/2, btn_y+btn_h/2-(fC.size if hasattr(fC,'size') else 18)/2), cta, font=fC, fill=brand)
                outs.append(np.array(bg.convert("RGB")))
            return outs
    """),
    f"{PKG}/services/feedback.py": textwrap.dedent("""\
        from typing import List, Dict
        def summarize_reviews(reviews: List[str]) -> str:
            return f"{len(reviews)} comentarios analizados. (Demo local)"
        def score_sentiment(reviews: List[str]) -> List[Dict]:
            out=[]; 
            for r in reviews:
                lab = "positivo" if any(x in r.lower() for x in ["bueno","excelente","me gusta"]) else "neutral"
                out.append({"review": r[:120], "sentiment": lab})
            return out
    """),
    f"{PKG}/services/feedback_gemini.py": textwrap.dedent("""\
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
            prompt=("Eres analista de CX. Resume en 3-5 bullets y 1 recomendación. Español conciso.\\n"
                    "REVIEWS_JSON:\\n"+json.dumps(reviews[:300], ensure_ascii=False))
            resp=c.models.generate_content(model=GEMINI_MODEL,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(prompt)])],
                    config=types.GenerateContentConfig(max_output_tokens=512, temperature=0.4))
            return (resp.text or "").strip()

        def score_sentiment_gemini(reviews: List[str]) -> List[Dict]:
            from google.genai import types
            c=_client()
            sys=('Clasifica cada review en positivo/negativo/neutral. Devuelve SOLO JSON: '
                 '[{"review":"","sentiment":"","rationale":""}]')
            prompt=sys+"\\nREVIEWS_JSON:\\n"+json.dumps(reviews[:200], ensure_ascii=False)
            resp=c.models.generate_content(model=GEMINI_MODEL,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(prompt)])],
                    config=types.GenerateContentConfig(max_output_tokens=2048, temperature=0.2))
            text=(resp.text or "").strip()
            try: return json.loads(text)
            except Exception:
                m=re.search(r"```(?:json)?\\s*(\\[[\\s\\S]*?\\])\\s*```", text, flags=re.I)
                if m: return json.loads(m.group(1))
                m2=re.search(r"(\\[[\\s\\S]*\\])", text)
                if m2: return json.loads(m2.group(1))
                return [{"review": r[:160], "sentiment":"neutral", "rationale":""} for r in reviews[:50]]
    """),

    # --- config/docs ---
    ".env.example": textwrap.dedent("""\
        # === API pública (rápido para desarrollo) ===
        GOOGLE_API_KEY=xxxxxx
        GEMINI_MODEL=gemini-2.5-flash-lite
        FORCE_GEMINI_PUBLIC=true

        # === Vertex (prod) ===
        # GCP_PROJECT=tu-proyecto
        # GCP_LOCATION=us-central1
        # GOOGLE_APPLICATION_CREDENTIALS=/ruta/credentiales.json
    """),
    "requirements.txt": textwrap.dedent("""\
        streamlit
        pandas
        python-dotenv
        pillow
        numpy
        google-genai>=0.3.0
        # Si usas Vertex Imagen:
        google-cloud-aiplatform>=1.70.0
    """),
    "README.md": "# GenAI MVP – Descripciones, Imágenes y Feedback\n\nEjecuta: `pip install -r requirements.txt` y `streamlit run app/app.py`.\nConfigura variables en `.env` (copiar `.env.example`).\n",
}

def ensure_dirs():
    for path in list(FILES.keys()):
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def write_files():
    for path, content in FILES.items():
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✓ creado: {path}")
        else:
            print(f"• existe: {path}")

if __name__ == "__main__":
    ensure_dirs()
    write_files()
    print("\nListo. Copia `.env.example` a `.env`, instala requirements y ejecuta:")
    print("  pip install -r requirements.txt")
    print("  streamlit run app/app.py")

