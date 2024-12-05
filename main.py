from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Charger le modèle et le tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model")

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        return {"error": "Veuillez fournir un texte pour la prédiction."}
    
    # Tokenisation
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
    
    return {"predicted_class": predicted_class}
