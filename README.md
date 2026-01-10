# **FactLens: Misinformation Detection**


## **Overview**

FactLens is a machine learning algrothim hosted in a web application designed to detect misinformation in online articles, journals, and research papers. It combines a supervised fake-news classifier with an unsupervised rumor-theme extractor to provide users with a detailed analysis of any input text. It is served as a Flask web app, displaying a text box where the user can input an article's text, then displaying whether it is real or fake, the model's probabilities, and top keywords.

Users can paste article text into the interface, and FactLens returns:

* Probability that the article is **real** or **fake**
* Cluster themes extracted from the text
* Cluster sizes
* Top terms characterizing each theme

---

## **Significance**

Information online is not guaranteed to be factual. FactLens addresses this issue by:

* Automatically detecting text that is likely false
* Identifying rumor clusters and themes
* Displaying meaningful probabilities and explanations

This tool can be used by students, teachers, researchers, or anyone who wants to quickly fact-check information.

---

## **Project Structure**

```
FactLens/
│
├── Data/          # Dataset files (.csv from Kaggle)
├── Models/        # Saved ML models
├── Webapp/        # Frontend + backend
│   ├── static/    # UI logic, JS, CSS
│   └── template/  # HTML frontend
│
├── app.py         # Main Flask API
└── train_models.py# ML pipeline for classifier + clustering
```

The Flask server runs via `app.py` at:

```
http://127.0.0.1:8000
```

---

## **Machine Learning Approach**

### **Fake News Classification (Supervised ML)**

Uses **TF-IDF** + **Logistic Regression** with hyperparameter tuning (GridSearchCV).

### **Rumor Theme Clustering (Unsupervised ML)**

Uses **TF-IDF**, **PCA**, and **K-Means (K=6)** to identify themes in misinformation.

---

## **Clustering & Analysis**

### **Dimensionality Reduction**

* PCA used to reduce feature dimensionality
* Preserves important patterns in the data

### **Clustering**

* Implemented **K-Means (K=6)**
* Groups similar text claims and misinformation patterns

### **Output**

* Identifies common patterns in misinformation
* Provides additional context alongside fake/real classification

### **Illustration**

```
Original Data → PCA Reduction → K-Means Clustered Data
```

---

## **Model Performance**

### **Evaluation Metrics**

* **Validation Accuracy:** 99%
* **Test Accuracy:** 99.14%
* **Precision (Fake class):** 0.9942
* **Recall:** 0.9893
* **F1-score:** 0.9917

### **Confusion Matrix**

| Actual \ Predicted | REAL | FAKE |
| ------------------ | ---- | ---- |
| **REAL**           | 3192 | 20   |
| **FAKE**           | 37   | 3405 |

These results indicate extremely strong model performance, with very low misclassification rates for both classes.

---

## **Data Collection**

Dataset sourced from:
**Kaggle Fake and Real News Dataset**
[https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Files stored under `/data/`.

---

## **Data Preprocessing & Feature Engineering**

### **Preprocessing Steps**

* Filling missing fields
* Merging title + text when needed
* Removing low-quality elements
* TF-IDF vectorization

### **Feature Engineering**

**For Classification:**

* TF-IDF with N-grams to capture richer context

**For Clustering:**

* TF-IDF vectors
* PCA for dimensionality reduction

---

## **Model Development**

### **Supervised Fake News Classifier**

* TF-IDF vectorization
* Logistic Regression
* Optimized via **GridSearchCV**
* Output includes probabilities for both real/fake

### **Unsupervised Clustering**

* TF-IDF → PCA → K-Means
* Extracted keywords help identify misinformation themes

---

## **Conclusion**

FactLens demonstrates an effective approach to combating misinformation by combining supervised classification with unsupervised clustering. The app analyzes article text, predicts whether it is real or fake, and highlights underlying themes.

**Future Improvements:**

* Larger and more diverse datasets


