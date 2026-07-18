import re
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# -------------------------------------------------
# File Paths
# -------------------------------------------------

FAKE_CSV = "dataset/Fake.csv"
TRUE_CSV = "dataset/True.csv"

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


# -------------------------------------------------
# Text Cleaning
# -------------------------------------------------

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


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

print("Loading dataset...")

fake_df = pd.read_csv(FAKE_CSV)
true_df = pd.read_csv(TRUE_CSV)

fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
) 

# -------------------------------------------------
# Prepare Content
# -------------------------------------------------

if "title" in df.columns and "text" in df.columns:
    df["content"] = (
        df["title"].fillna("") + " " + df["text"].fillna("")
    )
elif "text" in df.columns:
    df["content"] = df["text"].fillna("")
else:
    raise Exception("No text column found in dataset.")

print("Cleaning text...")

df["content"] = df["content"].apply(clean_text)

# Remove empty rows
df = df[df["content"].str.strip() != ""]

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# -------------------------------------------------
# Train Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# -------------------------------------------------
# TF-IDF Vectorizer
# -------------------------------------------------

print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -------------------------------------------------
# Train Model
# -------------------------------------------------

print("Training Logistic Regression...")

model = LogisticRegression(max_iter=1000)

model.fit(
    X_train_vec,
    y_train
) 

# -------------------------------------------------
# Evaluate Model
# -------------------------------------------------

print("Evaluating model...")

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -------------------------------------------------
# Save Model & Vectorizer
# -------------------------------------------------

print("Saving model...")

with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)

with open(VECTORIZER_PATH, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\n====================================")
print("Training Completed Successfully!")
print("Model saved to:", MODEL_PATH)
print("Vectorizer saved to:", VECTORIZER_PATH)
print("====================================")