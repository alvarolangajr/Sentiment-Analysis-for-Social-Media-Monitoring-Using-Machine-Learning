import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "training.1600000.processed.noemoticon.csv"

FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

columns = [
    "target",
    "id",
    "date",
    "query",
    "user",
    "text"
]

df = pd.read_csv(
    DATA_PATH,
    encoding="latin-1",
    header=None,
    names=columns
)

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nOriginal sentiment distribution:")
print(df["target"].value_counts())


# ============================================================
# 2. SELECT POSITIVE AND NEGATIVE CLASSES
# ============================================================

# Sentiment140:
# 0 = negative
# 4 = positive
#
# The original dataset also contains a neutral code (2),
# but this project focuses on binary classification.

df = df[df["target"].isin([0, 4])].copy()

# Convert labels:
# negative = 0
# positive = 1

df["sentiment"] = df["target"].map({
    0: 0,
    4: 1
})

print("\nBinary sentiment distribution:")
print(df["sentiment"].value_counts())


# ============================================================
# 3. CLEAN TEXT
# ============================================================

def clean_text(text):
    """
    Basic preprocessing for social-media text.

    Steps:
    - convert to lowercase
    - remove URLs
    - remove usernames/mentions
    - remove non-alphanumeric symbols
    - remove extra whitespace
    """

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning text...")

df["clean_text"] = df["text"].apply(clean_text)

print("\nExample original text:")
print(df["text"].iloc[0])

print("\nExample cleaned text:")
print(df["clean_text"].iloc[0])


# ============================================================
# 4. REMOVE EMPTY TEXT RECORDS
# ============================================================

df = df[df["clean_text"].str.len() > 0].copy()

print("\nDataset after cleaning:")
print(df.shape)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. TF-IDF FEATURE EXTRACTION
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    min_df=2
)

print("\nCreating TF-IDF features...")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ============================================================
# 7. DEFINE MACHINE-LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ),

    "Naive Bayes": MultinomialNB(),

    "SVM": LinearSVC(
        max_iter=1000
    )
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = {}
predictions = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    # Train model
    model.fit(X_train_tfidf, y_train)

    # Generate predictions
    y_pred = model.predict(X_test_tfidf)

    predictions[model_name] = y_pred

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Negative", "Positive"],
            zero_division=0
        )
    )


# ============================================================
# 9. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results).T

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)


# Save model results inside the project folder
results_df.to_csv(
    BASE_DIR / "model_results.csv",
    index=True
)


# ============================================================
# 10. CONFUSION MATRICES
# ============================================================

for model_name, y_pred in predictions.items():

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")

    # Make a safe filename
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300
    )

    plt.show()


# ============================================================
# 11. CLASSIFICATION ERROR ANALYSIS
# ============================================================

# Use the predictions from Logistic Regression
# for an example of examining incorrectly classified tweets.

error_analysis = pd.DataFrame({
    "text": X_test.values,
    "actual": y_test.values,
    "predicted": predictions["Logistic Regression"]
})

errors = error_analysis[
    error_analysis["actual"] != error_analysis["predicted"]
].copy()

print("\nNumber of misclassified tweets:")
print(len(errors))

print("\nExample classification errors:")
print(errors.head(20))


# Save classification errors inside the project folder
errors.to_csv(
    BASE_DIR / "classification_errors.csv",
    index=False
)


print("\nAnalysis complete.")