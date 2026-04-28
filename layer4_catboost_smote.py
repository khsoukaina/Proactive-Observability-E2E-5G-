import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support, accuracy_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

print("=== Layer 4: CatBoost with SMOTE + Stratified CV ===")

df = pd.read_csv("features/L3_zombie_features.csv")

# Inject regime variation
if df["regime"].nunique() == 1:
    df.loc[df["zombie_confirmed"] == 1, "regime"] = 3
    df.loc[(df["anomaly_score"] == -1) & (df["regime"] == 0), "regime"] = 2
    df.loc[(df["mod_req_rate"] > 0) & (df["regime"] == 0), "regime"] = 1

FEATURES = [
    "mod_req_rate", "mod_req_rate_delta", "acceleration",
    "session_churn_rate", "failure_ratio",
    "gbr_ratio", "voice_ratio", "weighted_criticality",
    "avg_weight", "throughput_bps",
    "session_delta", "gtpu_silence_ratio",
    "zombie_confirmed", "anomaly_raw_score",
    "unique_sessions", "total_pkts",
]
available = [f for f in FEATURES if f in df.columns]
X = df[available].fillna(0).values
y = df["regime"].fillna(0).astype(int).values

print(f"Original: {dict(zip(*np.unique(y, return_counts=True)))}")

# Filter classes with at least 2 samples (SMOTE needs k_neighbors+1)
class_counts = pd.Series(y).value_counts()
keep = class_counts[class_counts >= 2].index.tolist()
mask = np.isin(y, keep)
X, y = X[mask], y[mask]
print(f"After filtering: {dict(zip(*np.unique(y, return_counts=True)))}")

n_splits = min(3, min(np.bincount(y)[np.bincount(y) > 0]))
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_results = []
all_y_true, all_y_pred = [], []

print(f"\n=== {n_splits}-Fold CV with SMOTE on training fold only ===\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # SMOTE on training fold ONLY (no test contamination)
    train_counts = np.bincount(y_train)
    min_count = train_counts[train_counts > 0].min()
    if min_count >= 2:
        k = min(min_count - 1, 3)
        smote = SMOTE(k_neighbors=k, random_state=42)
        try:
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"  Fold {fold}: SMOTE failed ({e}), using original")
            X_train_bal, y_train_bal = X_train, y_train
    else:
        X_train_bal, y_train_bal = X_train, y_train

    model = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=4,
        loss_function="MultiClass", random_seed=42, verbose=0)
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test).flatten()
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)

    fold_results.append({
        "fold": fold, "n_train": len(X_train_bal), "n_test": len(y_test),
        "accuracy": acc, "precision": p, "recall": r, "f1": f1
    })
    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    print(f"  Fold {fold}: train={len(X_train_bal):3d} (after SMOTE), "
          f"test={len(y_test):3d}, acc={acc:.3f}, f1={f1:.3f}")

results_df = pd.DataFrame(fold_results)
print(f"\n=== SMOTE-Augmented CV Performance ===")
print(f"  Accuracy:  {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")
print(f"  Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}")
print(f"  Recall:    {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}")
print(f"  F1:        {results_df['f1'].mean():.3f} ± {results_df['f1'].std():.3f}")

print(f"\n=== Per-Class Performance (aggregated) ===")
print(classification_report(all_y_true, all_y_pred, zero_division=0))

print(f"\n=== Confusion Matrix ===")
cm = confusion_matrix(all_y_true, all_y_pred)
print(pd.DataFrame(cm,
    index=[f"True_{c}" for c in sorted(set(all_y_true))],
    columns=[f"Pred_{c}" for c in sorted(set(all_y_pred))]))

# Final model on full data with SMOTE
min_count = np.bincount(y)[np.bincount(y) > 0].min()
if min_count >= 2:
    k = min(min_count - 1, 5)
    X_final, y_final = SMOTE(k_neighbors=k, random_state=42).fit_resample(X, y)
else:
    X_final, y_final = X, y

final_model = CatBoostClassifier(
    iterations=300, learning_rate=0.05, depth=4,
    loss_function="MultiClass", random_seed=42, verbose=0)
final_model.fit(X_final, y_final)

importance = pd.DataFrame({
    "feature": available,
    "importance": final_model.get_feature_importance()
}).sort_values("importance", ascending=False)

importance.to_csv("features/L4_feature_importance.csv", index=False)
final_model.save_model("models/catboost_regime_classifier.cbm")
results_df.to_csv("features/L4_cv_results.csv", index=False)
df["predicted_regime"] = final_model.predict(df[available].fillna(0).values).flatten()
df.to_csv("features/L4_final_dataset.csv", index=False)

print(f"\nSaved:")
print(f"  features/L4_cv_results.csv")
print(f"  features/L4_feature_importance.csv")
print(f"  features/L4_final_dataset.csv")
print(f"  models/catboost_regime_classifier.cbm")
