from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold
)

from sklearn.neural_network import MLPClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5

# Each original image has 4 weather versions:
# daylight, rainy, snow, night
WEATHER_VARIANTS = 4


# ============================================================
# LOAD PCA-REDUCED DATA
# ============================================================

print("=" * 65)
print("MLP CLASSIFIER WITH STOCHASTIC GRADIENT DESCENT")
print("=" * 65)

X_train = np.load(
    DATA_DIR / "X_train_pca.npy"
)

y_train = np.load(
    DATA_DIR / "y_train.npy"
)

print(
    f"\nTraining data shape : {X_train.shape}"
)

print(
    f"Training labels     : {y_train.shape}"
)


# ============================================================
# BASIC CHECKS
# ============================================================

if X_train.shape[0] != y_train.shape[0]:
    raise ValueError(
        "Number of training samples and labels do not match."
    )

if not np.isfinite(X_train).all():
    raise ValueError(
        "X_train contains invalid values."
    )

print(
    "\nClasses:"
)

print(
    f"  Road Sign  (0): "
    f"{np.sum(y_train == 0)}"
)

print(
    f"  Pedestrian (1): "
    f"{np.sum(y_train == 1)}"
)


# ============================================================
# CREATE GROUP IDs
# ============================================================
#
# The preprocessing pipeline keeps the 4 weather versions
# of each original image together.
#
# Example:
#
# Original image 1:
#   daylight
#   rainy
#   snow
#   night
#
# All four receive the same group ID.
#
# This prevents weather versions of the same image from
# appearing in different cross-validation folds.
# ============================================================

if X_train.shape[0] % WEATHER_VARIANTS != 0:
    raise ValueError(
        "Training samples are not divisible by the "
        "number of weather variants."
    )

groups = (
    np.arange(X_train.shape[0])
    // WEATHER_VARIANTS
)

print(
    f"\nOriginal training groups : "
    f"{len(np.unique(groups))}"
)

print(
    f"Weather images per group  : "
    f"{WEATHER_VARIANTS}"
)


# ============================================================
# STRATIFIED GROUP CROSS-VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# BASE MLP
# ============================================================

mlp = MLPClassifier(
    activation="relu",
    solver="sgd",

    # Training control
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,

    random_state=RANDOM_STATE
)


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================
#
# We test:
#
# 1. Hidden-layer architecture
# 2. Learning rate
# 3. Momentum
#
# The final combination is selected using F1 score.
# ============================================================

parameter_grid = {

    "hidden_layer_sizes": [
        (32, 16),
        (64, 32),
        (128, 64)
    ],

    "learning_rate_init": [
        0.001,
        0.01
    ],

    "momentum": [
        0.80,
        0.90,
        0.95
    ]
}


# ============================================================
# GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=mlp,

    param_grid=parameter_grid,

    scoring="f1",

    cv=cv,

    n_jobs=-1,

    return_train_score=True,

    verbose=1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 65)
print("TRAINING MLP")
print("=" * 65)

print(
    "\nSearching for the best:"
)

print(
    "  • Hidden-layer architecture"
)

print(
    "  • Learning rate"
)

print(
    "  • Momentum"
)

print(
    "\nUsing 5-fold Stratified Group Cross-Validation..."
)

print(
    "Weather variants of the same original image "
    "remain in the same fold."
)


grid_search.fit(
    X_train,
    y_train,
    groups=groups
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_

best_parameters = grid_search.best_params_

best_cv_f1 = grid_search.best_score_


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("MLP TRAINING COMPLETE")
print("=" * 65)

print(
    "\nBest parameters:"
)

print(
    f"  Hidden layers    : "
    f"{best_parameters['hidden_layer_sizes']}"
)

print(
    f"  Learning rate    : "
    f"{best_parameters['learning_rate_init']}"
)

print(
    f"  Momentum         : "
    f"{best_parameters['momentum']}"
)

print(
    f"\nBest CV F1 score   : "
    f"{best_cv_f1:.4f}"
)

print(
    f"Training iterations: "
    f"{best_model.n_iter_}"
)

print(
    f"Final training loss: "
    f"{best_model.loss_:.6f}"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_DIR / "mlp_sgd.joblib"
)


# ============================================================
# SAVE CV RESULTS
# ============================================================

cv_results = grid_search.cv_results_

np.save(
    RESULTS_DIR / "mlp_cv_results.npy",
    cv_results,
    allow_pickle=True
)


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

with open(
    RESULTS_DIR / "mlp_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "MLP + SGD TRAINING SUMMARY\n"
    )

    file.write(
        "=" * 55 + "\n"
    )

    file.write(
        f"Training samples: "
        f"{X_train.shape[0]}\n"
    )

    file.write(
        f"PCA features: "
        f"{X_train.shape[1]}\n"
    )

    file.write(
        f"Cross-validation folds: "
        f"{N_SPLITS}\n"
    )

    file.write(
        f"Best hidden layers: "
        f"{best_parameters['hidden_layer_sizes']}\n"
    )

    file.write(
        f"Best learning rate: "
        f"{best_parameters['learning_rate_init']}\n"
    )

    file.write(
        f"Best momentum: "
        f"{best_parameters['momentum']}\n"
    )

    file.write(
        f"Best CV F1: "
        f"{best_cv_f1:.6f}\n"
    )

    file.write(
        f"Training iterations: "
        f"{best_model.n_iter_}\n"
    )

    file.write(
        f"Final training loss: "
        f"{best_model.loss_:.6f}\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)

print(
    "Model saved to:"
)

print(
    "  models/mlp_sgd.joblib"
)

print(
    "\nResults saved to:"
)

print(
    "  results/mlp_summary.txt"
)

print(
    "  results/mlp_cv_results.npy"
)

print(
    "\nMLP + SGD training completed successfully."
)

print("=" * 65)