import joblib

# -----------------------------
# Load TF-IDF Vectorizer
# -----------------------------
tfidf_vectorizer = joblib.load(r"C:\Users\Godwin\Documents\OneDrive\Desktop\review\models\tfidf_vectorizer (1).pkl")

# -----------------------------
# Load Model B
# -----------------------------
model_b = joblib.load(r"C:\Users\Godwin\Documents\OneDrive\Desktop\review\models\ModelB.pkl")

# -----------------------------
# Example new texts
# -----------------------------
new_texts = [
    "This product is amazing and works great!",
    "Terrible experience, I will not buy again."
]

# Convert text to TF-IDF features
X_new = tfidf_vectorizer.transform(new_texts)

# Predict numeric ratings
y_pred_num = model_b.predict(X_new)

# -----------------------------
# Print results
# -----------------------------
for review, rating in zip(new_texts, y_pred_num):
    print(f"Review: {review}")
    print(f"Predicted Rating: {rating}\n")

# -----------------------------
# Optional: User input prediction
# -----------------------------
user_input = input("Enter a review to predict rating: ")
X_user = tfidf_vectorizer.transform([user_input])
y_user_pred = model_b.predict(X_user)
print("Predicted Rating:", y_user_pred[0])
