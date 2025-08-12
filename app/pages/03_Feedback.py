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
