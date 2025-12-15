# Toxic Comment Classifier 🚦

A simple project to detect toxic comments using both classical ML and DistilBERT models.

---

## 🔹 Features

- **Model Training Notebook**: Preprocess dataset, train TF-IDF+LogReg & DistilBERT, save models.
- **Backend API**: `/predict` endpoint (FastAPI) for toxicity classification.
- **Frontend**: Minimal HTML form for live predictions.
- **Deployment Ready**: One-click run with `python app.py`.

---

## 🔹 Installation

```bash
# Clone the repo and move into the directory
git clone <repo-url>
cd <repo-directory>

# Install dependencies
pip install -r requirements.txt
```

---

## 🔹 How to Train

Edit and run `train_model.ipynb` (Jupyter Notebook).  
- Saves: `models/tfidf_logreg.pkl`, `models/distilbert.pt`

---

## 🔹 How to Run Backend & Frontend

```bash
# Start the FastAPI server
python app.py
```
- The API will be at: [http://localhost:8000/docs](http://localhost:8000/docs)
- The frontend UI will be at: [http://localhost:8000](http://localhost:8000)

---

## 🔹 Example Screenshots

```
[Insert screenshots here, e.g., model training, prediction UI, API response]
```

---

## 🔹 File Structure

```
project-root/
  ├─ requirements.txt
  ├─ README.md
  ├─ train_model.ipynb
  ├─ app.py
  ├─ models/
  ├─ frontend/
  └─ utils/
```