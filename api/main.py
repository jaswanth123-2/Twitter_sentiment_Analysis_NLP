from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from typing import List

app = FastAPI(title="Sentiment Analysis API")

model = None
tokenizer = None
device = None

class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

@app.on_event("startup")
async def load_model():
    global model, tokenizer, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.load_state_dict(torch.load('../models/checkpoints/best_roberta_final.pt', map_location=device))
    model.to(device)
    model.eval()

def predict(text: str):
    encoding = tokenizer.encode_plus(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence

@app.get("/")
def root():
    return {
        "status": "running",
        "model": "RoBERTa Sentiment Classifier",
        "version": "1.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: TextRequest):
    try:
        sentiment, confidence = predict(request.text)
        return PredictionResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    try:
        results = []
        for text in request.texts:
            sentiment, confidence = predict(text)
            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence
            })
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))