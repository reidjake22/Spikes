"""
Nested Cross-Validation Logistic Regression Pipeline with Feature Caching

This script implements a robust nested cross-validation pipeline to select the optimal L₂
regularization parameter for multinomial logistic regression on preprocessed MNIST-like datasets.
Features are cached to avoid re-pivoting.

- Caches prepare_features outputs in ./cache/<dataset>_features.npz
- Nested cross-validation (5 outer folds, 3 inner folds).
- Two-stage hyperparameter tuning: coarse grid + adaptive randomized search.
- Parallel CPU execution and logging progress.

Dependencies:
- numpy, pandas, scikit-learn, joblib
"""

import os


os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Data extraction and preprocessing utilities
import data_extraction as de

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Globals ─────────────────────────────────────────────────────────────────
SEED = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
COARSE_C_GRID = np.logspace(0, 4, 9)
N_RANDOM_SEARCH = 20
CACHE_DIR = "cache"

type(de)
outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)

base_estimator = LogisticRegression(solver="lbfgs", penalty="l2",
    max_iter=2000, tol=1e-4, random_state=SEED
)

# ── Hyperparameter Tuning ────────────────────────────────────────────────────
def tune_hyperparameters(X, y):
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", base_estimator)])
    logger.info("  [Inner CV] Starting coarse grid search")

    coarse = GridSearchCV(
        pipeline, {"clf__C": COARSE_C_GRID},
        cv=inner_cv, scoring="accuracy", n_jobs=1,verbose=100000
    )
    coarse.fit(X, y)
    C_coarse = coarse.best_params_["clf__C"]
    logger.info(f"  [Inner CV] Coarse best C = {C_coarse:.4g}")

    lower, upper = C_coarse / 10, C_coarse * 10
    logger.info(f"  [Inner CV] Starting adaptive search in [{lower:.4g}, {upper:.4g}]")
    adaptive = RandomizedSearchCV(
        pipeline,
        {"clf__C": np.logspace(np.log10(lower), np.log10(upper), N_RANDOM_SEARCH)},
        n_iter=N_RANDOM_SEARCH, cv=inner_cv, scoring="accuracy",
        random_state=SEED, n_jobs=-1
    )
    adaptive.fit(X, y)
    C_refined = adaptive.best_params_["clf__C"]
    best_score = adaptive.best_score_
    logger.info(f"  [Inner CV] Refined best C = {C_refined:.4g} (score = {best_score:.4f})")
    return C_refined, best_score

# ── Nested CV Evaluation ─────────────────────────────────────────────────────
def nested_cv_evaluation(X, y, dataset_name, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    records = []
    logger.info(f"[Outer CV] Dataset={dataset_name}, samples={len(y)}, features={X.shape[1]}")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        logger.info(f"[Outer CV] Fold {fold}/{N_OUTER_FOLDS}: tuning")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        C_opt, inner_acc = tune_hyperparameters(X_train, y_train)

        logger.info(f"[Outer CV] Fold {fold}: retraining with C={C_opt:.4g}")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                multi_class="multinomial", solver="lbfgs", penalty="l2",
                C=C_opt, max_iter=2000, tol=1e-4, random_state=SEED
            ))
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"[Outer CV] Fold {fold}: test_acc={acc:.4f}, test_loss={loss:.4f}")

        records.append({
            "dataset": dataset_name,
            "fold": fold,
            "C": C_opt,
            "inner_accuracy": inner_acc,
            "test_accuracy": acc,
            "test_log_loss": loss
        })

        cm_file = os.path.join(save_dir, f"cm_{dataset_name}_fold{fold}.npy")
        np.save(cm_file, cm)

    df_res = pd.DataFrame(records)
    out_csv = os.path.join(save_dir, f"nested_cv_{dataset_name}.csv")
    df_res.to_csv(out_csv, index=False)
    logger.info(f"Nested CV complete for {dataset_name}. Results → {out_csv}")
    logger.info(f"Summary:\n{df_res.describe()}")

# ── Main Pipeline ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    weight_versions = ["epoch_0_item_0", "epoch_0_item_60000"]
    noise_versions = ["noise_0", "noise_5"]
    split = "train"

    for wv in weight_versions:
        for nv in noise_versions:
            name = f"{wv}_{nv}"
            logger.info(f"=== Processing dataset: {name} ===")


            # 2) Feature caching
            cache_path = os.path.join(CACHE_DIR, f"{name}_features.npz")
            if os.path.exists(cache_path):
                logger.info(f"Loading cached features from {cache_path}")
                data = np.load(cache_path)
                X, y = data["X"], data["y"]
            else:
                # 1) Load raw data
                files = de.get_parquet_files(wv, nv, split)
                df = de.load_and_combine_parquet(files)

                logger.info("Preparing features fresh")
                X, y = de.prepare_features(df, max_stimuli=None)
                logger.info(f"Caching features to {cache_path}")
                np.savez(cache_path, X=X, y=y)
            logger.info(f"scaling features for dataset {name}: {X.shape[0]} samples, {X.shape[1]} features")
            # 3) Scale features
            X_scaled = de.scale_features(X)
            logger.info(f"Features scaled: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
            # 4) Nested CV training & evaluation
            nested_cv_evaluation(X_scaled, y, name, save_dir=os.path.join("results", name))
