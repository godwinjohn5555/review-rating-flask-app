# app.py
import os
import pickle
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load Models & Vectorizers
# -------------------------
BASE_DIR = BASE_DIR = os.path.join(os.path.dirname(__file__), "models")
 #r"C:\Users\Godwin\Documents\OneDrive\Desktop\review\models"

# Model A (FNN + TF-IDF)
with open(os.path.join(BASE_DIR, "vectorizer_modelA.pkl"), "rb") as f:
    tokenizer_a = pickle.load(f)  # TF-IDF vectorizer

model_a = tf.keras.models.load_model(os.path.join(BASE_DIR, "ModelA.keras"))

# Create LabelEncoder inside code (no external file needed)
label_encoder_a = LabelEncoder()
label_encoder_a.fit([1, 2, 3, 4, 5])  # Ratings are always 1–5

# Model B (sklearn + TF-IDF)
vectorizer_b = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer (1).pkl"))
model_b = joblib.load(os.path.join(BASE_DIR, "ModelB.pkl"))

# -------------------------
# Prediction Functions
# -------------------------
def predict_model_a(text):
    """
    Predict using Model A (FNN + TF-IDF, classification mode)
    Converts prediction back to rating (1–5)
    Returns 'Not Available' if input has no valid tokens
    """
    X = tokenizer_a.transform([text])

    # If TF-IDF produced an empty vector -> no valid words
    if X.nnz == 0:  # nnz = number of non-zero entries
        return "Not Available"

    try:
        pred = model_a.predict(X)
        predicted_class = np.argmax(pred, axis=1)       # 0–4
        rating = label_encoder_a.inverse_transform(predicted_class)[0]  # 1–5
        return int(rating)
    except Exception as e:
        print("Model A error:", e)
        return "Not Available"


def predict_model_b(text):
    """
    Predict using Model B (sklearn + TF-IDF)
    Returns 'Not Available' if input has no valid tokens
    """
    X = vectorizer_b.transform([text])

    if X.nnz == 0:
        return "Not Available"

    try:
        pred = model_b.predict(X)
        return int(pred[0])  # already 1–5
    except Exception as e:
        print("Model B error:", e)
        return "Not Available"


# -------------------------
# Flask Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form["review"]

    # Get predictions
    rating_a = predict_model_a(user_input)
    rating_b = predict_model_b(user_input)

    return render_template(
        "result.html",
        review=user_input,
        rating_a=rating_a,
        rating_b=rating_b
    )

# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
