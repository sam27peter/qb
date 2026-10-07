from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Class 1 = Pedestrian
# Class 0 = Road Sign

CLASS_NAMES = [
    "Road Sign",
    "Pedestrian"
]


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 65)
print("MODEL EVALUATION")
print("=" * 65)

X_test = np.load(
    DATA_DIR / "X_test_pca.npy"
)

y_test = np.load(
    DATA_DIR / "y_test.npy"
)

print(
    f"\nTest data shape : {X_test.shape}"
)

print(
    f"Test labels     : {y_test.shape}"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model_path = MODEL_DIR / "mlp_sgd.joblib"

if not model_path.exists():
    raise FileNotFoundError(
        "Trained model not found: "
        f"{model_path}"
    )

model = joblib.load(
    model_path
)

print(
    f"Loaded model: {model_path}"
)


# ============================================================
# CHECK DATA
# ============================================================

if X_test.shape[0] != y_test.shape[0]:
    raise ValueError(
        "Number of test samples and labels do not match."
    )

if not np.isfinite(X_test).all():
    raise ValueError(
        "X_test contains invalid values."
    )


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test
)

# Probability of class 1 = Pedestrian
y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)


# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# PRECISION-RECALL / AVERAGE PRECISION
# ============================================================

average_precision = average_precision_score(
    y_test,
    y_probability
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("TEST SET RESULTS")
print("=" * 65)

print(
    f"\nAccuracy          : {accuracy:.4f}"
)

print(
    f"Precision         : {precision:.4f}"
)

print(
    f"Recall            : {recall:.4f}"
)

print(
    f"F1 Score          : {f1:.4f}"
)

print(
    f"ROC-AUC           : {roc_auc:.4f}"
)

print(
    f"Average Precision : {average_precision:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

print(
    report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print(
    "\n              Predicted"
)

print(
    "              Road  Pedestrian"
)

print(
    f"Actual Road     {cm[0, 0]:4d}      {cm[0, 1]:4d}"
)

print(
    f"Actual Ped.     {cm[1, 0]:4d}      {cm[1, 1]:4d}"
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

np.save(
    RESULTS_DIR / "confusion_matrix.npy",
    cm
)


# ============================================================
# GRAPH 1 — CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    CLASS_NAMES
)

plt.yticks(
    [0, 1],
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)


# Add numbers inside the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )


plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2 — ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    y_probability
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"MLP-SGD (AUC = {roc_auc:.4f})"
)

# Random classifier reference
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "roc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3 — PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_probability
    )
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_values,
    precision_values,
    linewidth=2,
    label=f"MLP-SGD (AP = {average_precision:.4f})"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "precision_recall_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "average_precision": average_precision
}


np.save(
    RESULTS_DIR / "evaluation_metrics.npy",
    metrics,
    allow_pickle=True
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

with open(
    RESULTS_DIR / "evaluation_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "MLP + SGD TEST EVALUATION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "Test samples: "
        f"{len(y_test)}\n"
    )

    file.write(
        "PCA features: "
        f"{X_test.shape[1]}\n\n"
    )

    file.write(
        f"Accuracy          : {accuracy:.6f}\n"
    )

    file.write(
        f"Precision         : {precision:.6f}\n"
    )

    file.write(
        f"Recall            : {recall:.6f}\n"
    )

    file.write(
        f"F1 Score          : {f1:.6f}\n"
    )

    file.write(
        f"ROC-AUC           : {roc_auc:.6f}\n"
    )

    file.write(
        f"Average Precision : "
        f"{average_precision:.6f}\n\n"
    )

    file.write(
        "CONFUSION MATRIX\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\nCLASSIFICATION REPORT\n"
    )

    file.write(
        report
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.save(
    RESULTS_DIR / "y_test_predictions.npy",
    y_pred
)

np.save(
    RESULTS_DIR / "y_test_probabilities.npy",
    y_probability
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)
print("EVALUATION COMPLETED")
print("=" * 65)

print("\nGenerated files:")

print(
    "  results/confusion_matrix.png"
)

print(
    "  results/roc_curve.png"
)

print(
    "  results/precision_recall_curve.png"
)

print(
    "  results/evaluation_report.txt"
)

print(
    "  results/evaluation_metrics.npy"
)

print(
    "  results/y_test_predictions.npy"
)

print(
    "  results/y_test_probabilities.npy"
)

print("\nEvaluation completed successfully.")