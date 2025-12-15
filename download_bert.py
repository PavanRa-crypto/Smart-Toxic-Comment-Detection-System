from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

MODEL_NAME = "unitary/toxic-bert" # Reliable multi-label toxicity model
SAVE_PATH = "models/distilbert"

os.makedirs("models", exist_ok=True) 

print(f"Downloading model {MODEL_NAME}...")
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Save to your local directory
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"Successfully downloaded and saved multi-label model to {SAVE_PATH}")