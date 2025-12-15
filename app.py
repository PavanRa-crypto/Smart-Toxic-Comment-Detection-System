from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import os
import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import numpy as np 

# =====================
# Configuration & Setup
# =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Toxicity Detection and Transformation")

TOXICITY_LABELS = [
    "Toxic", "Severe Toxic", "Obscene", 
    "Threat", "Insult", "Identity Hate"
]

# =====================
# Frontend Setup
# =====================
templates = None
if os.path.isdir("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
    templates = Jinja2Templates(directory="frontend")
    logger.info("Frontend templates loaded.")
else:
    logger.warning("Frontend folder not found. Templates will not work.")

# =====================
# Load Models
# =====================
vec, clf = None, None
try:
    VEC_PATH = "models/tfidf_vectorizer.pkl"
    CLF_PATH = "models/tfidf_logreg.pkl"
    if os.path.isfile(VEC_PATH) and os.path.isfile(CLF_PATH):
        vec = joblib.load(VEC_PATH)
        clf = joblib.load(CLF_PATH)
        logger.info("TF-IDF model loaded successfully.")
    else:
        logger.warning("TF-IDF model files not found.")
except Exception as e:
    logger.error(f"Failed to load TF-IDF model: {e}")

bert_tokenizer, bert_model, device = None, None, "cpu"
try:
    DISTILBERT_PATH = "models/distilbert"
    if os.path.isdir(DISTILBERT_PATH):
        # 🔑 FINAL FIX: Use AutoTokenizer/AutoModel to infer the correct architecture (6 labels)
        bert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
        bert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_PATH) 
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_model.to(device)
        bert_model.eval() # Set model to evaluation mode
        logger.info(f"BERT model loaded on {device} using Auto classes.")
    else:
        logger.warning("BERT model folder not found.")
except Exception as e:
    logger.error(f"Failed to load BERT model: {e}")


# =====================
# Prediction Core Functions
# =====================

def get_multi_scores(text: str, model_type: str) -> Dict[str, float]:
    """
    Returns probabilities (0.0 to 1.0). 
    For BERT, returns all 6 actual multi-label scores. 
    For TF-IDF, returns only the single 'Toxic' score.
    """
    
    # 1. BERT (Multi-Label Logic)
    if model_type == "bert" and bert_model and bert_tokenizer:
        try:
            inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
                
                # Check for 6 logits (multi-label model output)
                if outputs.logits.size(1) == len(TOXICITY_LABELS):
                    # Apply Sigmoid to logits to get probabilities
                    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
                    
                    # Map the 6 probabilities to the 6 TOXICITY_LABELS
                    score_dict = {label: float(np.clip(probs[i], 0.0, 1.0))
                                  for i, label in enumerate(TOXICITY_LABELS)}
                    
                    return score_dict
                else:
                    logger.error(f"BERT model returned {outputs.logits.size(1)} logits, expected 6. Check model loading.")
                    return {label: 0.0 for label in TOXICITY_LABELS}

        except Exception as e:
            logger.error(f"BERT prediction error: {e}")
            return {label: 0.0 for label in TOXICITY_LABELS}

    # 2. TF-IDF (Binary Logic)
    elif model_type == "tfidf" and vec and clf:
        try:
            X = vec.transform([text])
            # Only get the probability of the 'toxic' class (index 1)
            prob_toxic = clf.predict_proba(X)[0][1]
            
            # Return only the single score, which will be simulated later
            return {"Toxic": float(prob_toxic)}
            
        except Exception as e:
            logger.error(f"TF-IDF prediction error: {e}")
            return {"Toxic": 0.0}
            
    return {"Toxic": 0.0} # Default if model is unavailable


def process_scores_for_ui(comment: str, model: str) -> Dict[str, Any]:
    """Calculates scores (either actual or simulated) and formats them for the Jinja2 template."""
    
    raw_scores = get_multi_scores(comment, model)
    
    results_data = {"scores": [], "comment": comment, "model": model}
    
    SUMMARY_THRESHOLD = 0.50 
    
    detected_labels = []
    is_overall_toxic = False
    
    def get_color_class(score: float) -> str:
        """Determines the color class for the score text."""
        if score >= 0.5:
            return "score-high"
        elif score >= 0.2:
            return "score-medium"
        else:
            return "score-low"
    
    # --- CORE LOGIC: Use Actual Multi-Label Scores for BERT ---
    
    # 1. If BERT, use the 6 actual predicted scores from get_multi_scores
    if model == "bert" and len(raw_scores) == len(TOXICITY_LABELS):
        final_scores = raw_scores
        
    # 2. If TF-IDF, use the original simulation logic
    else:
        prob_toxic = raw_scores.get("Toxic", 0.0)
        
        # Simulate other scores based on the core 'Toxic' probability
        final_scores = {
            "Toxic": prob_toxic,
            "Severe Toxic": prob_toxic * 0.25, 
            "Obscene": prob_toxic * 0.70,
            "Threat": prob_toxic * 0.15,
            "Insult": prob_toxic * 0.60,
            "Identity Hate": prob_toxic * 0.20
        }
    
    # --- Format Scores and Determine Status ---
    
    for label in TOXICITY_LABELS:
        score = final_scores.get(label, 0.0)
        score_percent = round(score * 100, 2)
        
        # Determine if the comment is overall toxic by checking if ANY category hits the threshold
        if score >= SUMMARY_THRESHOLD:
            detected_labels.append(label)
            is_overall_toxic = True
            
        results_data["scores"].append({
            "label": label,
            "score": score_percent,
            "color_class": get_color_class(score)
        })
        
    # Determine Summary Banner Message
    if is_overall_toxic:
        # Create a detailed message listing all detected categories
        if detected_labels:
             results_data["summary_message"] = "Toxic: " + ", ".join(detected_labels)
        else:
             results_data["summary_message"] = "Status: Toxic"
        
        results_data["summary_class"] = "toxic-content-detected"
    else:
        results_data["summary_message"] = "Status: Safe"
        results_data["summary_class"] = "clean-comment"
        
    return results_data

# =====================
# API Schema (Pydantic models remain the same)
# =====================
class PredictRequest(BaseModel):
    comment: str = Field(..., example="You are awesome!")
    model: Optional[str] = Field("bert", example="bert")

# =====================
# Routes
# =====================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if templates:
        # Pass empty results_data to render empty form
        return templates.TemplateResponse("index.html", {"request": request, "results_data": None})
    return HTMLResponse("<h1>Frontend not found</h1>")

@app.post("/", response_class=HTMLResponse)
async def classify_form(
    request: Request,
    comment: str = Form(...),
    model: str = Form("bert")
):
    model = model.lower()
    
    # Process scores using the simulation function
    results_data = process_scores_for_ui(comment, model)
    
    if templates:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request, 
                "results_data": results_data, 
            }
        )
    return HTMLResponse(f"<p>{comment} → {results_data['summary_message']} ({model})</p>")

@app.post("/predict", response_class=JSONResponse)
async def predict_api(payload: PredictRequest):
    comment = payload.comment
    model = (payload.model or "bert").lower()
    
    # Process scores using the simulation function
    scores_data = process_scores_for_ui(comment, model)
    
    # Return only the dictionary of scores (labels and percentage)
    return {item['label']: item['score'] for item in scores_data['scores']}