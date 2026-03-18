import os
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

RESULTS_DIR = "results/cnn_lstm_real_enriched_sampled"

train_errors = np.load(os.path.join(RESULTS_DIR, "train_errors.npy"))
test_errors = np.load(os.path.join(RESULTS_DIR, "test_errors.npy"))
y_test = np.load(os.path.join(RESULTS_DIR, "y_test.npy"))

percentiles = [99.9, 99.7, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0, 95.0]

best = None
all_results = []

print("Searching best threshold...\n")

for p in percentiles:
    threshold = np.percentile(train_errors, p)
    y_pred = (test_errors > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    result = {
        "percentile": p,
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }
    all_results.append(result)

    print(
        f"Percentile={p:5.1f} | "
        f"Threshold={threshold:.8f} | "
        f"Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}"
    )

    if best is None or f1 > best["f1"]:
        best = result

print("\nBest result:")
print(best)

with open(os.path.join(RESULTS_DIR, "threshold_search_results.json"), "w") as f:
    json.dump(all_results, f, indent=4)

with open(os.path.join(RESULTS_DIR, "best_threshold.json"), "w") as f:
    json.dump(best, f, indent=4)

best_threshold = best["threshold"]
y_pred_best = (test_errors > best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_best)

print("\nBest confusion matrix:")
print(cm)