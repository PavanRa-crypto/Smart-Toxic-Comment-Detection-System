# 🛡️ Smart Toxic Comment Detection System

A web-based machine learning application that detects **toxic and abusive comments** using both **classical NLP techniques (TF-IDF + Logistic Regression)** and **Transformer-based models (BERT)**.

This project demonstrates an **end-to-end ML pipeline**, from data preprocessing and model training to deployment with a web interface.

---

## 📌 Project Overview

Online platforms often face challenges in moderating toxic content such as hate speech, insults, and threats.
This system helps automatically identify such content by classifying user input into multiple toxicity categories.

The application supports **multi-label classification**, meaning a single comment can belong to more than one toxic category.

---

## ✨ Key Features

* 🔹 Multi-label toxicity detection
* 🔹 Two model options:

  * **TF-IDF + Logistic Regression** (fast & lightweight)
  * **BERT (Transformer-based model)**
* 🔹 Web-based UI for live prediction
* 🔹 Probability scores for each toxicity category
* 🔹 Clean and responsive frontend
* 🔹 FastAPI-based backend

---

## 🧠 Toxicity Categories Detected

* Toxic
* Severe Toxic
* Obscene
* Threat
* Insult
* Identity Hate

Each category is predicted independently with a confidence score.

---

## 🗂️ Project Structure

```
Smart-Toxic-Comment-Detection-System/
│
├── frontend/
│   ├── index.html
│   └── style.css
│
├── utils/
│   └── preprocess.py
│
├── models/                  # Generated after training
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_logreg.pkl
│
├── app.py                   # FastAPI application
├── train_model.ipynb        # Model training notebook
├── download_bert.py         # Script to download BERT model
├── train.csv                # Dataset (local, ignored by git)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset Information

* The system is **based on the Toxic Comment Classification dataset** (originally by Jigsaw / Kaggle).
* For demonstration and local testing, a **small curated dataset** is used.
* Large datasets are **not pushed to GitHub** to keep the repository clean and lightweight.

> 📌 *In real-world deployment, large-scale datasets such as Kaggle’s Toxic Comment dataset are used for training.*

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/PavanRa-crypto/Smart-Toxic-Comment-Detection-System.git
cd Smart-Toxic-Comment-Detection-System
```

---

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
```

**Activate:**

* Windows:

```bash
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If running manually:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib torch transformers python-multipart notebook
```

---

## 🏋️ Model Training

Open the training notebook:

```bash
jupyter notebook
```

Run all cells in:

```
train_model.ipynb
```

This will generate:

```
models/
├── tfidf_vectorizer.pkl
└── tfidf_logreg.pkl
```

---

## 🚀 Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open in browser:

* 🌐 Web UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* 📘 API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🖥️ Sample Output

* User enters a comment
* Model predicts toxicity probabilities
* System displays:

  * Overall status (Safe / Toxic)
  * Category-wise confidence scores

---

## 🖼️ Screenshots

Screenshots included in this project demonstrate:

* Home page UI
* Toxic comment prediction
* Safe comment prediction
* Model selection (TF-IDF / BERT)
* Running server (Uvicorn)

*(Screenshots can be found in the project report and documentation.)*

---

## 🎓 Academic Relevance

This project is suitable for:

* MCA final year project
* Machine Learning / NLP coursework
* Demonstration of ML deployment concepts

It covers:

* Text preprocessing
* Feature extraction (TF-IDF)
* Supervised learning
* Transformer models
* Web deployment using FastAPI

---

## 👨‍💻 Project Credits

**Developed by:**
**M. Pavan**
Master of Computer Applications (MCA)

**Project Title:**
**Smart Toxic Comment Detection System**

---

## 📜 License

This project is developed for **academic and educational purposes**.

---

### ✅ FINAL NOTE

This README is **submission-ready**.
Do **not** change model versions or retrain unless required.

---


