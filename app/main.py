from fastapi import FastAPI
from pydantic import BaseModel
from app.sentiment import analyze_sentiment

app = FastAPI()

class SentimentIn(BaseModel):
    text: str

@app.post("/sentiment")
def sentiment(inp: SentimentIn):
    return analyze_sentiment(inp.text)