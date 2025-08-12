from typing import List, Dict
def summarize_reviews(reviews: List[str]) -> str:
    return f"{len(reviews)} comentarios analizados. (Demo local)"
def score_sentiment(reviews: List[str]) -> List[Dict]:
    out=[]; 
    for r in reviews:
        lab = "positivo" if any(x in r.lower() for x in ["bueno","excelente","me gusta"]) else "neutral"
        out.append({"review": r[:120], "sentiment": lab})
    return out
