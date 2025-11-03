import json
import random
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import os

# ==============================
# Utility Functions
# ==============================

def compute_metrics(y_true, y_pred, save_path=None):
    """Compute accuracy, precision, recall, F1, and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {save_path}")

    return metrics


# ==============================
# Load Yes/No Subset
# ==============================

def load_yesno_data(split="test", num_samples=None):
    """Load yes/no subset of PathVQA dataset."""
    dataset = load_dataset("flaviagiammarino/path-vqa", split=split)
    data = [x for x in dataset if x["answer"].strip().lower() in ["yes", "no"]]

    if num_samples:
        data = data[:num_samples]

    questions = [x["question"] for x in data]
    labels = [1 if x["answer"].lower().strip() == "yes" else 0 for x in data]
    return questions, np.array(labels)


# ==============================
# Baseline 1 — Random Guesser
# ==============================

def random_baseline(y_true):
    """Randomly predict yes/no with equal probability."""
    y_pred = np.random.randint(0, 2, size=len(y_true))
    return compute_metrics(y_true, y_pred, save_path="metrics/random_baseline.json")


# ==============================
# Baseline 2 — Majority Class
# ==============================

def majority_baseline(y_true):
    """Always predict the most frequent label in the dataset."""
    majority_label = int(np.round(np.mean(y_true)))  # 1 if yes majority, else 0
    y_pred = np.full_like(y_true, fill_value=majority_label)
    return compute_metrics(y_true, y_pred, save_path="metrics/majority_baseline.json")


# ==============================
# Baseline 3 — Logistic Regression (Text Only)
# ==============================

def logistic_regression_baseline(questions, y_true, num_samples=None):
    """Train and evaluate a text-only logistic regression baseline."""
    if num_samples:
        questions = questions[:num_samples]
        y_true = y_true[:num_samples]

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    print("Encoding questions with DistilBERT...")
    features = []
    with torch.no_grad():
        for q in tqdm(questions, desc="Encoding"):
            inputs = tokenizer(q, return_tensors="pt", truncation=True, padding=True, max_length=32)
            outputs = model(**inputs)
            sentence_emb = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            features.append(sentence_emb)
    X = np.vstack(features)

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y_true)
    y_pred = clf.predict(X)

    return compute_metrics(y_true, y_pred, save_path="metrics/logreg_baseline.json")


# ==============================
# Run All Baselines
# ==============================

def run_all_baselines(split="test", num_samples=None):
    print(f"\nRunning baselines on split='{split}' with num_samples={num_samples}")
    questions, y_true = load_yesno_data(split, num_samples)

    print("\n1️. Random Guesser")
    random_results = random_baseline(y_true)
    print(random_results)

    print("\n2️. Majority-Class Predictor")
    majority_results = majority_baseline(y_true)
    print(majority_results)

    print("\n3️. Logistic Regression (Text Only)")
    logreg_results = logistic_regression_baseline(questions, y_true, num_samples)
    print(logreg_results)

    print("\nAll baselines completed.")
    return {
        "random": random_results,
        "majority": majority_results,
        "logistic_regression": logreg_results,
    }


if __name__ == "__main__":
    # Example usage:
    # Run on a small subset to test quickly
    results = run_all_baselines(split="validation", num_samples=200)
