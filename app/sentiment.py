from transformers import pipeline
from functools import lru_cache 

@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    # Small, fast, CPU friendly
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
def analyze_sentiment(text: str) -> dict:
    clf = get_sentiment_pipeline()
    res = clf(text)[0]
    return {"label": res["label"], "score": float(res["score"])}