from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le modèle et le tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model")

# Texte à tester
text = "Quand les consultations médicales à distance sauvent des vies humaines. Babyl Rwanda est une application mobile qui aide des citoyens rwandais et étrangers résidant au Rwanda à consulter un médecin à distance. Depuis 2016, plus de 2.900.000 personnes ont utilisé les services pour traiter plusieurs maladies, telles que le paludisme, les"

# Tokenisation
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Prédiction
outputs = model(**inputs)
logits = outputs.logits
predicted_class = torch.argmax(logits).item()

print("Classe prédite :", predicted_class)
