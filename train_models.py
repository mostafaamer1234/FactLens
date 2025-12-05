
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib



RANDOM_STATE = 42
DATA_DIR = "data"
MODELS_DIR = "models"

NEWS_CSV_PATH = os.path.join(DATA_DIR, "news.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def ensure_news_csv_exists():


    if os.path.exists(NEWS_CSV_PATH):
        print(f">>> Using existing {NEWS_CSV_PATH}")
        return

    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if os.path.exists(fake_path) and os.path.exists(true_path):
        print(">>> Creating news.csv from Fake.csv + True.csv…")

        fake = pd.read_csv(fake_path)
        true = pd.read_csv(true_path)

        fake["label"] = "FAKE"
        true["label"] = "REAL"

        fake = fake[["title", "text", "label"]]
        true = true[["title", "text", "label"]]

        df = pd.concat([fake, true], ignore_index=True)
        df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

        df.to_csv(NEWS_CSV_PATH, index=False)
        print(f">>> Saved merged dataset to {NEWS_CSV_PATH} (rows: {len(df)})")

        return

    raise RuntimeError(
        "ERROR: Could not find data/news.csv or data/Fake.csv and data/True.csv.\n\n"
        "Please download the Kaggle dataset manually:\n"
        "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n\n"
        "Then place Fake.csv and True.csv into the 'data/' folder:\n"
        "  project/\n"
        "    data/\n"
        "      Fake.csv\n"
        "      True.csv\n"
        "    train_models.py\n\n"
        "After that, run: python train_models.py"
    )


# -------------------------------------------------------------
# STEP 2: LOAD + CLEAN DATA
# -------------------------------------------------------------

def load_and_clean_data(path):
    print(f">>> Loading dataset from {path}...")
    df = pd.read_csv(path)

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df["text_full"] = df["title"] + " " + df["text"]

    df = df[(df["text_full"].str.len() > 100) & (df["text_full"].str.len() < 20000)]

    # Convert labels to 0/1
    df["label"] = df["label"].map({"REAL": 0, "FAKE": 1}).astype(int)

    print(">>> Cleaned dataset shape:", df.shape)
    print(">>> Label counts:\n", df["label"].value_counts())

    return df




def train_classifier(df):
    X = df["text_full"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
            )
        ),
        (
            "logreg",
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        )
    ])

    param_grid = {
        "tfidf__max_features": [5000, 10000],
        "logreg__C": [0.5, 1.0, 2.0],
    }

    print(">>> Running GridSearchCV for best classifier…")

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(">>> Best parameters:", grid.best_params_)

    print("\n>>> Validation results:")
    print(classification_report(y_valid, best_model.predict(X_valid)))

    print("\n>>> Test results:")
    print(classification_report(y_test, best_model.predict(X_test)))

    # Save model
    joblib.dump(best_model, os.path.join(MODELS_DIR, "fake_news_classifier.joblib"))
    print(">>> Saved classifier to models/fake_news_classifier.joblib")

    return best_model




def train_rumor_clusters(df, model, n_clusters=6):
    print(">>> Training rumor clusters (KMeans + PCA)…")

    fake_df = df[df["label"] == 1]
    fake_texts = fake_df["text_full"]

    tfidf = model.named_steps["tfidf"]
    X_fake_tfidf = tfidf.transform(fake_texts)

    # PCA for dimensionality reduction before clustering
    pca = PCA(n_components=50, random_state=RANDOM_STATE)
    X_fake_pca = pca.fit_transform(X_fake_tfidf.toarray())

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init="auto"
    )
    clusters = kmeans.fit_predict(X_fake_pca)

    # Save models
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_50.joblib"))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "rumor_kmeans.joblib"))

    print(">>> Saved PCA + KMeans models.")

    # Extract keywords per cluster
    terms = np.array(tfidf.get_feature_names_out())
    cluster_keywords = {}

    for c in range(n_clusters):
        indices = np.where(clusters == c)[0]
        if len(indices) == 0:
            cluster_keywords[c] = {"top_terms": [], "size": 0}
            continue

        means = X_fake_tfidf[indices].mean(axis=0).A1
        top_indices = means.argsort()[::-1][:15]

        cluster_keywords[c] = {
            "top_terms": terms[top_indices].tolist(),
            "size": int(len(indices)),
        }

    with open(os.path.join(MODELS_DIR, "cluster_keywords.json"), "w") as f:
        json.dump(cluster_keywords, f, indent=2)

    print(">>> Saved cluster_keywords.json")


def main():
    print("\n=== Starting FactLens Training ===\n")

    ensure_news_csv_exists()
    df = load_and_clean_data(NEWS_CSV_PATH)

    clf = train_classifier(df)
    train_rumor_clusters(df, clf)

    print("\n=== Training Complete! Models saved in /models ===\n")


if __name__ == "__main__":
    main()
