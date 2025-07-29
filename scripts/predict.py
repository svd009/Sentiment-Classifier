from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer from saved directory
model_path = "./models"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text: str):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # Run prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs).item()
    
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    confidence = probs[0][predicted_class].item()

    return {"sentiment": sentiment, "confidence": round(confidence, 4)}

if __name__ == "__main__":
    sample_text = "I absolutely loved this movie! The performances were stunning."
    print(predict_sentiment(sample_text))
