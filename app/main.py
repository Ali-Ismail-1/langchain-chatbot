from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.sentiment import analyze_sentiment
from app.rag import ask_with_context

app = FastAPI(title="FastAPI + LangChain Chatbot")

# ---------- Models ----------
class SentimentIn(BaseModel):
    text: str

class ChatIn(BaseModel):
    session_id: str
    message: str

# ---------- Routes ----------
@app.get("/")
def root():
    return {"ok": True, "service": "fastapi-langchain-chatbot"}

@app.post("/sentiment")
def sentiment(inp: SentimentIn):
    try:
        return analyze_sentiment(inp.text)
    except Exception as e:
        raise HTTPException(500, f"Sentiment error: {e}")

@app.post("/chat")
def chat(inp: ChatIn):
    try:
        answer = ask_with_context(inp.session_id, inp.message)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(500, f"Chat error: {e}")
