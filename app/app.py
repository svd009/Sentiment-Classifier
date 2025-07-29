from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# Load model & tokenizer
model_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Initialize FastAPI app
app = FastAPI(title="Sentiment Classifier API")

# Request schema
class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to the Sentiment Classifier API"}

@app.post("/predict")
def predict_sentiment(req: TextRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs).item()

    sentiment = "Positive" if predicted_class == 1 else "Negative"
    confidence = probs[0][predicted_class].item()

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }
