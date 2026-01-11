import os
import random
import re
from collections import Counter, defaultdict
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Data load
# ----------------------------

DATA_URL = "https://raw.githubusercontent.com/cbannard/lela60331_24-25/refs/heads/main/coursework/Compiled_Reviews.txt"
DATA_FILE = "Compiled_Reviews.txt"

reviews=[]
sentiment_ratings=[]
product_types=[]
helpfulness_ratings=[]

with open("Compiled_Reviews.txt") as f:
   for line in f.readlines()[1:]:
        fields = line.rstrip().split('\t')
        reviews.append(fields[0])
        sentiment_ratings.append(fields[1])
        product_types.append(fields[2])
        helpfulness_ratings.append(fields[3])

# ----------------------------
# Constants
# ----------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MAX_VOCAB = 10000
PUNCT_RE = re.compile(r"[^\w\s]")

# ----------------------------
# Mapping labels
# ----------------------------

def map_helpfulness_label(lbl):
    string = lbl.lower()
    if string.startswith("help"):
        return "helpful"
    if string.startswith("unhelp"):
        return "unhelpful"
    return "neutral"

def map_sentiment_label(lbl):
    string = lbl.lower()
    if string.startswith("pos"):
        return "positive"
    if string.startswith("neg"):
        return "negative"
    return "neutral"

# ----------------------------
# Text preprocessing
# ----------------------------

def preprocess(text, use_bigrams=True, use_trigrams=False):
    text = PUNCT_RE.sub(" ", text.lower())
    tokens = text.split()

    features = list(tokens)

    if use_bigrams:
        features.extend(make_ngrams(tokens, 2))

    return features

def make_ngrams(tokens, n):
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# ----------------------------
# Vectorisation
# ----------------------------

def build_vocab(reviews, indices):
    counter = Counter()
    for i in indices:
        counter.update(preprocess(reviews[i]))
    return {w: i for i, (w, _) in enumerate(counter.most_common(MAX_VOCAB))}

def vectorise_reviews(reviews, indices, vocab):
    X = np.zeros((len(indices), len(vocab)), dtype=np.float32)
    for j, i in enumerate(indices):
        for tok in preprocess(reviews[i]):
            if tok in vocab:
                X[j, vocab[tok]] += 1.0
    return X

# ----------------------------
# Logistic regression (mini-batch SGD)
# ----------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def binary_cross_entropy(y, p): 
    eps = 1e-12 
    p = np.clip(p, eps, 1 - eps) 
    return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))

def train_logistic_regression(
    X_tr, y_tr, X_dev, y_dev,
    lr=0.05, epochs=500, batch_size=256, l2=1e-6
):
    w = np.zeros(X_tr.shape[1], dtype=np.float32)
    b = 0.0
    n = len(X_tr)
    history = {"train_loss": [], "dev_loss": []}

    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            Xb, yb = X_tr[idx], y_tr[idx]

            p = predict_proba(Xb, w, b)
            err = p - yb

            grad_w = (Xb.T @ err) / len(Xb) + l2 * w
            grad_b = err.mean()

            w -= lr * grad_w
            b -= lr * grad_b

        # record losses end of epoch
        p_tr_full = predict_proba(X_tr, w, b)
        p_dev_full = predict_proba(X_dev, w, b)
        history["train_loss"].append(float(binary_cross_entropy(y_tr, p_tr_full)))
        history["dev_loss"].append(float(binary_cross_entropy(y_dev, p_dev_full)))

    return w, b, history

# ----------------------------
# Utils
# ----------------------------

def stratified_split(labels, dev_frac=0.1, test_frac=0.2):
    groups = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[map_helpfulness_label(lbl)].append(i)

    splits = {"train": [], "dev": [], "test": []}
    for idxs in groups.values():
        random.shuffle(idxs)
        n = len(idxs)
        t = int(n * test_frac)
        d = int(n * dev_frac)
        splits["test"] += idxs[:t]
        splits["dev"] += idxs[t:t + d]
        splits["train"] += idxs[t + d:]

    for k in splits:
        random.shuffle(splits[k])
    return splits

def labels_A(indices):
    return np.array(
        [map_helpfulness_label(helpfulness_ratings[i]) == "helpful" for i in indices],
        dtype=np.float32
    )

def labels_B(indices):
    return np.array(
        [map_helpfulness_label(helpfulness_ratings[i]) != "unhelpful" for i in indices],
        dtype=np.float32
    )

def sentiment_labels(indices):
    return np.array(
        [map_sentiment_label(sentiment_ratings[i]) == "positive" for i in indices],
        dtype=np.float32
    )

def classification_metrics(y_true, y_pred): 
   tp = np.sum((y_true == 1) & (y_pred == 1)) 
   tn = np.sum((y_true == 0) & (y_pred == 0)) 
   fp = np.sum((y_true == 0) & (y_pred == 1)) 
   fn = np.sum((y_true == 1) & (y_pred == 0)) 
   acc = (tp + tn) / max(1, tp + tn + fp + fn) 
   prec = tp / max(1, tp + fp) 
   rec = tp / max(1, tp + fn) 
   return { 
       "accuracy": acc, 
       "precision": prec, 
       "recall": rec, 
       "tp": int(tp), 
       "tn": int(tn), 
       "fp": int(fp), 
       "fn": int(fn) 
       }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----------------------------
# Main pipeline !!!
# ----------------------------
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_pipeline():
    splits = stratified_split(helpfulness_ratings)

    # ----- Usefulness models -----
    vocab = build_vocab(reviews, splits["train"])
    X_tr = vectorise_reviews(reviews, splits["train"], vocab)
    X_dev = vectorise_reviews(reviews, splits["dev"], vocab)
    X_te = vectorise_reviews(reviews, splits["test"], vocab)

    yA_tr, yA_dev = labels_A(splits["train"]), labels_A(splits["dev"])
    yB_tr, yB_dev = labels_B(splits["train"]), labels_B(splits["dev"])

    wA, bA, histA = train_logistic_regression(X_tr, yA_tr, X_dev, yA_dev)
    wB, bB, histB = train_logistic_regression(X_tr, yB_tr, X_dev, yB_dev)

    usefulness = predict_proba(X_te, wA, bA) + predict_proba(X_te, wB, bB)

    # ----- Sentiment model -----
    sent_vocab = build_vocab(reviews, splits["train"])
    Xs_tr = vectorise_reviews(reviews, splits["train"], sent_vocab)
    Xs_dev = vectorise_reviews(reviews, splits["dev"], sent_vocab)
    Xs_te = vectorise_reviews(reviews, splits["test"], sent_vocab)

    ys_tr = sentiment_labels(splits["train"])
    ys_dev = sentiment_labels(splits["dev"])

    ws, bs, histS = train_logistic_regression(Xs_tr, ys_tr, Xs_dev, ys_dev)
    sentiment_probs = predict_proba(Xs_te, ws, bs)

    # ----------------------------
    # Model evaluation on Test
    # ----------------------------

    # ----- Usefulness model A -----
    yA_te = labels_A(splits["test"])
    yA_pred = (predict_proba(X_te, wA, bA) >= 0.5).astype(int)

    metrics_A = classification_metrics(yA_te, yA_pred)

    print("\nModel A (Helpful vs not) metrics:")
    for k, v in metrics_A.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # ----- Usefulness model B -----
    yB_te = labels_B(splits["test"])
    yB_pred = (predict_proba(X_te, wB, bB) >= 0.5).astype(int)

    metrics_B = classification_metrics(yB_te, yB_pred)

    print("\nModel B (Not-unhelpful vs unhelpful) metrics:")
    for k, v in metrics_B.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # ----- Sentiment model -----
    ys_te = sentiment_labels(splits["test"])
    ys_pred = (sentiment_probs >= 0.5).astype(int)

    metrics_S = classification_metrics(ys_te, ys_pred)

    print("\nSentiment model metrics:")
    for k, v in metrics_S.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")



    # ----- Correlation -----
    corr = np.corrcoef(sentiment_probs, usefulness)[0, 1]
    print(f"\nSentiment–usefulness correlation: {corr:.3f}")

    # ----- Usefulness histogram -----
    print("\nUsefulness histogram:")
    bins = np.arange(0, 2.1, 0.1)
    counts, edges = np.histogram(usefulness, bins=bins)
    for i, c in enumerate(counts):
        print(f"[{edges[i]:.1f}, {edges[i+1]:.1f}): {c}")

    # Visualise usefulness histogram with matplotlib
    plt.figure(figsize=(7, 4))
    plt.hist(usefulness, bins=bins, edgecolor="black")
    plt.xlabel("Usefulness score")
    plt.ylabel("Number of reviews")
    plt.title("Usefulness score distribution")
    plt.tight_layout()
    plt.show()

    # ----- Product means -----
    by_product = defaultdict(list)
    for j, i in enumerate(splits["test"]):
        by_product[product_types[i]].append(usefulness[j])

    means = {k: float(np.mean(v)) for k, v in by_product.items()}
    print("\nTop 5 product types:", sorted(means.items(), key=lambda x: -x[1])[:5])
    print("Bottom 5 product types:", sorted(means.items(), key=lambda x: x[1])[:5])

      # ----------------------------
      # extremity - usefulness analysis
      # ----------------------------

      # Sentiment extremity per review
    sentiment_extremity = np.abs(sentiment_probs - 0.5)

      # by prod
    by_product_ext = defaultdict(list)
    by_product_use = defaultdict(list)

    for j, i in enumerate(splits["test"]):
       prod = product_types[i]
       by_product_ext[prod].append(sentiment_extremity[j])
       by_product_use[prod].append(usefulness[j])

    product_means = []

    for prod in by_product_ext:
       mean_ext = float(np.mean(by_product_ext[prod]))
       mean_use = float(np.mean(by_product_use[prod]))
       product_means.append((prod, mean_ext, mean_use))

      # Convert to arrays for correlation
    mean_extremities = np.array([x[1] for x in product_means], dtype=np.float32)
    mean_usefulness = np.array([x[2] for x in product_means], dtype=np.float32)

      # Correlation
    prod_level_corr = np.corrcoef(mean_extremities, mean_usefulness)[0, 1]

    print("\nProduct-level correlation (mean extremity vs mean usefulness):",
          round(float(prod_level_corr), 3))

      # Strongest / weakest categories
    print("\nTop 5 products by mean sentiment extremity:")
    print(sorted(product_means, key=lambda x: -x[1])[:5])

    print("\nBottom 5 products by mean sentiment extremity:")
    print(sorted(product_means, key=lambda x: x[1])[:5])



    # ----------------------------
    # Binned sentiment - usefulness curve
    # ----------------------------
    bins = np.linspace(0, 1, 11)
    ids = np.digitize(sentiment_probs, bins) - 1

    xs, ys = [], []
    for b in range(len(bins) - 1):
        mask = ids == b
        if np.any(mask):
            xs.append((bins[b] + bins[b+1]) / 2)
            ys.append(np.mean(usefulness[mask]))

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Sentiment probability (positive)")
    plt.ylabel("Mean usefulness")
    plt.title("Usefulness vs sentiment polarity")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Sentiment extremity
    # ----------------------------
    extremity = np.abs(sentiment_probs - 0.5)

    bins = np.linspace(0, 0.5, 11)
    ids = np.digitize(extremity, bins) - 1

    xs, ys = [], []
    for b in range(len(bins) - 1):
        mask = ids == b
        if np.any(mask):
            xs.append((bins[b] + bins[b+1]) / 2)
            ys.append(np.mean(usefulness[mask]))

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Sentiment extremity |p - 0.5|")
    plt.ylabel("Mean usefulness")
    plt.title("Usefulness vs sentiment extremity")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # True helpfulness counts
    # ----------------------------

    print("\nTEST SET helpfulness distribution:")
    true_labels = [map_helpfulness_label(helpfulness_ratings[i]) for i in splits["test"]]
    counts = Counter(true_labels)
    total = sum(counts.values())
    for k in ["helpful", "neutral", "unhelpful"]:
        print(f"  {k}: {counts[k]} ({counts[k]/total:.3f})")

    # ----------------------------
    # Mean usefulness by true label
    # ----------------------------

    by_label = defaultdict(list)
    for j, i in enumerate(splits["test"]):
        lbl = map_helpfulness_label(helpfulness_ratings[i])
        by_label[lbl].append(usefulness[j])

    print("\nMean usefulness by true helpfulness label:")
    for lbl in ["unhelpful", "neutral", "helpful"]:
        vals = by_label[lbl]
        print(f"  {lbl}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, n={len(vals)}")

    # ----------------------------
    # Overlayed histogram
    # ----------------------------

    bins = np.arange(0, 2.1, 0.1)

    plt.figure(figsize=(7,4))
    plt.hist(by_label["unhelpful"], bins=bins, alpha=0.5, label="unhelpful")
    plt.hist(by_label["neutral"], bins=bins, alpha=0.5, label="neutral")
    plt.hist(by_label["helpful"], bins=bins, alpha=0.5, label="helpful")

    plt.xlabel("Usefulness score")
    plt.ylabel("Count")
    plt.title("Usefulness distribution by true helpfulness label")
    plt.legend()
    plt.tight_layout()
    plt.show()



# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    run_pipeline()