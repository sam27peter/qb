from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

VARIANCE_THRESHOLD = 0.95
RANDOM_STATE = 42

# Number of components shown in the Scree Plot
SCREE_COMPONENTS = 30


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PCA - PRINCIPAL COMPONENT ANALYSIS")
print("=" * 60)

X_train = np.load(DATA_DIR / "X_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")

original_features = X_train.shape[1]

print(f"Original features per image : {original_features}")


# ============================================================
# DATA CHECK
# ============================================================

if not np.isfinite(X_train).all():
    raise ValueError("X_train contains invalid values.")

if not np.isfinite(X_test).all():
    raise ValueError("X_test contains invalid values.")


# ============================================================
# FIT PCA
# ============================================================

print("\nFitting PCA...")
print("Target explained variance:", VARIANCE_THRESHOLD)

pca = PCA(
    n_components=VARIANCE_THRESHOLD,
    svd_solver="full",
    random_state=RANDOM_STATE
)

# Fit PCA ONLY on training data
X_train_pca = pca.fit_transform(X_train)

# Apply the same PCA to test data
X_test_pca = pca.transform(X_test)


# ============================================================
# PCA INFORMATION
# ============================================================

selected_components = pca.n_components_

explained_variance_ratio = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(
    explained_variance_ratio
)


# ============================================================
# FIND COMPONENTS FOR DIFFERENT VARIANCE LEVELS
# ============================================================

components_85 = (
    np.searchsorted(cumulative_variance, 0.85) + 1
)

components_90 = (
    np.searchsorted(cumulative_variance, 0.90) + 1
)

components_95 = (
    np.searchsorted(cumulative_variance, 0.95) + 1
)


# ============================================================
# DIMENSION REDUCTION
# ============================================================

reduction_percentage = (
    1 -
    selected_components / original_features
) * 100


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PCA RESULTS")
print("=" * 60)

print(
    f"Original features       : {original_features}"
)

print(
    f"85% variance             : "
    f"{components_85} components"
)

print(
    f"90% variance             : "
    f"{components_90} components"
)

print(
    f"95% variance             : "
    f"{components_95} components"
)

print(
    f"Selected components     : "
    f"{selected_components}"
)

print(
    f"Final explained variance : "
    f"{cumulative_variance[-1]:.4f}"
)

print(
    f"Feature reduction       : "
    f"{reduction_percentage:.2f}%"
)

print(
    f"\nX_train PCA shape        : "
    f"{X_train_pca.shape}"
)

print(
    f"X_test PCA shape         : "
    f"{X_test_pca.shape}"
)


# ============================================================
# SAVE PCA MODEL
# ============================================================

joblib.dump(
    pca,
    MODEL_DIR / "pca.joblib"
)


# ============================================================
# SAVE PCA DATA
# ============================================================

np.save(
    DATA_DIR / "X_train_pca.npy",
    X_train_pca
)

np.save(
    DATA_DIR / "X_test_pca.npy",
    X_test_pca
)

np.save(
    RESULTS_DIR / "pca_explained_variance.npy",
    cumulative_variance
)

np.save(
    RESULTS_DIR / "pca_component_variance.npy",
    explained_variance_ratio
)


# ============================================================
# GRAPH 1 — SCREE PLOT
# ============================================================

scree_count = min(
    SCREE_COMPONENTS,
    len(explained_variance_ratio)
)

scree_components = np.arange(
    1,
    scree_count + 1
)

scree_values = explained_variance_ratio[
    :scree_count
]

plt.figure(figsize=(10, 6))

plt.plot(
    scree_components,
    scree_values,
    linestyle=":",
    marker="o",
    markersize=5,
    linewidth=1.5
)

plt.xlabel(
    "Principal Component Number"
)

plt.ylabel(
    "Individual Explained Variance"
)

plt.title(
    "PCA Scree Plot"
)

plt.xticks(
    np.arange(1, scree_count + 1, 2)
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "pca_scree_plot.png",
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 2 — CUMULATIVE EXPLAINED VARIANCE
# ============================================================

components = np.arange(
    1,
    len(cumulative_variance) + 1
)

plt.figure(figsize=(10, 6))

plt.plot(
    components,
    cumulative_variance,
    linewidth=2
)

plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "PCA Cumulative Explained Variance"
)

plt.ylim(
    0.30,
    1.02
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "pca_cumulative_variance.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 85% LINE
# ------------------------------------------------------------

plt.axhline(
    y=0.85,
    linestyle="--",
    linewidth=1.5,
    label=f"85% → {components_85} components"
)


# ------------------------------------------------------------
# 90% LINE
# ------------------------------------------------------------

plt.axhline(
    y=0.90,
    linestyle="--",
    linewidth=1.5,
    label=f"90% → {components_90} components"
)


# ------------------------------------------------------------
# 95% LINE
# ------------------------------------------------------------

plt.axhline(
    y=0.95,
    linestyle="--",
    linewidth=1.5,
    label=f"95% → {components_95} components"
)


# ------------------------------------------------------------
# MARK 85%
# ------------------------------------------------------------

plt.scatter(
    components_85,
    cumulative_variance[components_85 - 1],
    s=70,
    zorder=5
)


plt.annotate(
    f"{components_85} components\n85%",
    (
        components_85,
        cumulative_variance[components_85 - 1]
    ),
    xytext=(10, -35),
    textcoords="offset points"
)


# ------------------------------------------------------------
# MARK 90%
# ------------------------------------------------------------

plt.scatter(
    components_90,
    cumulative_variance[components_90 - 1],
    s=70,
    zorder=5
)


plt.annotate(
    f"{components_90} components\n90%",
    (
        components_90,
        cumulative_variance[components_90 - 1]
    ),
    xytext=(10, -35),
    textcoords="offset points"
)


# ------------------------------------------------------------
# MARK 95%
# ------------------------------------------------------------

plt.scatter(
    components_95,
    cumulative_variance[components_95 - 1],
    s=80,
    zorder=5
)


plt.annotate(
    f"{components_95} components\n95%",
    (
        components_95,
        cumulative_variance[components_95 - 1]
    ),
    xytext=(-100, -45),
    textcoords="offset points"
)


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "PCA Cumulative Explained Variance"
)

plt.ylim(
    0.30,
    1.02
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "pca_cumulative_variance.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    RESULTS_DIR / "pca_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("PCA SUMMARY\n")
    file.write("=" * 50 + "\n")

    file.write(
        f"Original features: "
        f"{original_features}\n"
    )

    file.write(
        f"85% variance: "
        f"{components_85} components\n"
    )

    file.write(
        f"90% variance: "
        f"{components_90} components\n"
    )

    file.write(
        f"95% variance: "
        f"{components_95} components\n"
    )

    file.write(
        f"Selected components: "
        f"{selected_components}\n"
    )

    file.write(
        f"Final explained variance: "
        f"{cumulative_variance[-1]:.6f}\n"
    )

    file.write(
        f"Feature reduction: "
        f"{reduction_percentage:.2f}%\n"
    )

    file.write(
        f"X_train PCA shape: "
        f"{X_train_pca.shape}\n"
    )

    file.write(
        f"X_test PCA shape: "
        f"{X_test_pca.shape}\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\nFiles saved successfully:")

print(
    "  models/pca.joblib"
)

print(
    "  data/processed/X_train_pca.npy"
)

print(
    "  data/processed/X_test_pca.npy"
)

print(
    "  results/pca_scree_plot.png"
)

print(
    "  results/pca_cumulative_variance.png"
)

print(
    "  results/pca_component_variance.npy"
)

print(
    "  results/pca_explained_variance.npy"
)

print(
    "  results/pca_summary.txt"
)

print("\nPCA completed successfully.")