import re
import pickle

from flask import Flask, render_template, request

# -----------------------------
# Flask App
# -----------------------------

app = Flask(__name__)

# -----------------------------
# Load Model & Vectorizer
# -----------------------------

with open("models/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# -----------------------------
# Text Cleaning
# -----------------------------

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove Reuters/AP/AFP datelines
    text = re.sub(
        r"^[A-Z][A-Za-z\s]+\((Reuters|AP|AFP)\)\s*-\s*",
        "",
        text,
    )

    # Remove wire-service names
    text = re.sub(
        r"\b(Reuters|Associated Press|AFP)\b",
        "",
        text,
    )

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        news=""
    )


# -----------------------------
# Prediction Route
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form.get("news", "").strip()

    if news == "":
        return render_template(
            "index.html",
            prediction="Please enter some news text.",
            confidence=None,
            news=""
        )

    cleaned_news = clean_text(news)

    transformed_news = vectorizer.transform([cleaned_news])

    prediction = model.predict(transformed_news)[0]

    probabilities = model.predict_proba(transformed_news)[0]

    classes = list(model.classes_)

    predicted_index = classes.index(prediction)

    confidence = probabilities[predicted_index] * 100

    if prediction == "REAL":
        result = "✅ REAL NEWS"
    else:
        result = "❌ FAKE NEWS"

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        news=news
    )
# -----------------------------
# Run the Flask App
# -----------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )