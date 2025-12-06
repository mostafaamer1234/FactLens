import os
import json

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# PATH SETUP

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# LOAD MODELS

classifier_path = os.path.join(MODELS_DIR, "fake_news_classifier.joblib")
pca_path = os.path.join(MODELS_DIR, "pca_50.joblib")
kmeans_path = os.path.join(MODELS_DIR, "rumor_kmeans.joblib")
keywords_path = os.path.join(MODELS_DIR, "cluster_keywords.json")

if not all(os.path.exists(p) for p in [classifier_path, pca_path, kmeans_path, keywords_path]):
    raise RuntimeError(
        "Models not found. Make sure you ran train_models.py and that the 'models/' "
        "directory contains fake_news_classifier.joblib, pca_50.joblib, rumor_kmeans.joblib, "
        "and cluster_keywords.json."
    )

clf = joblib.load(classifier_path)
pca_50 = joblib.load(pca_path)
kmeans = joblib.load(kmeans_path)

with open(keywords_path, "r", encoding="utf-8") as f:
    CLUSTER_KEYWORDS = json.load(f)

tfidf = clf.named_steps["tfidf"]
logreg = clf.named_steps["logreg"]

# Label mapping: 1 -> FAKE, 0 -> REAL
LABEL_MAP = {1: "FAKE", 0: "REAL"}

app = Flask(__name__)


# HELPER FUNCTIONS

def classify_article(text: str):
    """Return predicted label (0/1), probabilities, and human-readable label."""
    probs = clf.predict_proba([text])[0]  # order corresponds to clf.classes_
    classes = clf.classes_

    # find index of label "1" (FAKE)
    fake_idx = int(np.where(classes == 1)[0][0])
    real_idx = int(np.where(classes == 0)[0][0])

    fake_prob = float(probs[fake_idx])
    real_prob = float(probs[real_idx])

    pred_label_int = int(clf.predict([text])[0])
    pred_label_str = LABEL_MAP[pred_label_int]

    return {
        "label_int": pred_label_int,
        "label_str": pred_label_str,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
    }


def assign_cluster(text: str):
    """Assign cluster id and return associated keywords."""
    X_tfidf = tfidf.transform([text])
    X_pca50 = pca_50.transform(X_tfidf.toarray())
    cluster_id = int(kmeans.predict(X_pca50)[0])

    info = CLUSTER_KEYWORDS.get(str(cluster_id)) or CLUSTER_KEYWORDS.get(cluster_id)
    if info is None:
        info = {"top_terms": [], "size": 0}

    top_terms = info.get("top_terms", [])
    size = info.get("size", 0)

    # Create a simple name like "Cluster 0: politics, election, vote"
    if top_terms:
        name = f"Cluster {cluster_id}: " + ", ".join(top_terms[:5])
    else:
        name = f"Cluster {cluster_id}"

    return {
        "cluster_id": cluster_id,
        "cluster_name": name,
        "top_terms": top_terms,
        "size": size,
    }


# ROUTES

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    article_text = request.form.get("article_text", "").strip()

    if not article_text:
        return jsonify({"error": "No article text provided."}), 400

    cls = classify_article(article_text)

    cl = assign_cluster(article_text)

    response = {
        "label": cls["label_str"],
        "label_int": cls["label_int"],
        "fake_probability": cls["fake_prob"],
        "real_probability": cls["real_prob"],
        "cluster_id": cl["cluster_id"],
        "cluster_name": cl["cluster_name"],
        "cluster_top_terms": cl["top_terms"],
        "cluster_size": cl["size"],
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
