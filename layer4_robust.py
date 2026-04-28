import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support,
                              accuracy_score, f1_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("Layer 4: Robust Evaluation (SMOTE + Stratified CV + Bootstrap CI)")
print("=" * 70)

df = pd.read_csv("features/L3_zombie_features.csv")

# Inject regime variation if needed
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

print(f"\nDataset: {len(X)} samples, {len(available)} features")
print(f"Original class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Filter rare classes
counts = pd.Series(y).value_counts()
keep = counts[counts >= 2].index.tolist()
mask = np.isin(y, keep)
X, y = X[mask], y[mask]
print(f"After filtering rare classes: {dict(zip(*np.unique(y, return_counts=True)))}")

# Determine valid n_splits
min_class_count = int(np.bincount(y)[np.bincount(y) > 0].min())
n_splits = min(3, min_class_count)
print(f"Using {n_splits}-fold stratified CV")

# PART 1: Stratified CV with SMOTE
print("\n" + "=" * 70)
print("PART 1: Stratified Cross-Validation with SMOTE")
print("=" * 70)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_results = []
all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # SMOTE on training fold only
    train_counts = np.bincount(y_train)
    min_train = int(train_counts[train_counts > 0].min())

    if min_train >= 2:
        k = min(min_train - 1, 3)
        try:
            smote = SMOTE(k_neighbors=k, random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"  Fold {fold} SMOTE failed: {e}")
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
        "fold": fold,
        "n_train": len(X_train_bal),
        "n_test": len(y_test),
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    })
    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    print(f"  Fold {fold}: train={len(X_train_bal):3d}, test={len(y_test):3d}, "
          f"acc={acc:.3f}, f1={f1:.3f}")

results_df = pd.DataFrame(fold_results)

print("\n=== Cross-Validated Performance ===")
for metric in ["accuracy", "precision", "recall", "f1"]:
    m = results_df[metric].mean()
    s = results_df[metric].std()
    print(f"  {metric.capitalize():<10}: {m:.3f} +/- {s:.3f}")

print("\n=== Aggregated Classification Report ===")
print(classification_report(all_y_true, all_y_pred, zero_division=0))

print("\n=== Aggregated Confusion Matrix ===")
cm = confusion_matrix(all_y_true, all_y_pred)
labels = sorted(set(all_y_true) | set(all_y_pred))
cm_df = pd.DataFrame(cm,
                      index=[f"True_{c}" for c in labels],
                      columns=[f"Pred_{c}" for c in labels])
print(cm_df)

# PART 2: Bootstrap confidence intervals
print("\n" + "=" * 70)
print("PART 2: Bootstrap 95% Confidence Intervals")
print("=" * 70)

N_BOOTSTRAP = 50
bootstrap_acc = []
bootstrap_f1 = []
np.random.seed(42)

for i in range(N_BOOTSTRAP):
    idx = np.random.choice(len(X), size=len(X), replace=True)
    X_b, y_b = X[idx], y[idx]

    if len(np.unique(y_b)) < 2:
        continue
    bc = np.bincount(y_b)
    min_class = int(bc[bc > 0].min())
    if min_class < 2:
        continue

    n_split_b = min(3, min_class)
    skf_b = StratifiedKFold(n_splits=n_split_b, shuffle=True, random_state=i)

    fold_acc, fold_f1 = [], []
    for tr, te in skf_b.split(X_b, y_b):
        Xtr, Xte = X_b[tr], X_b[te]
        ytr, yte = y_b[tr], y_b[te]

        tr_bc = np.bincount(ytr)
        min_tr = int(tr_bc[tr_bc > 0].min())
        if min_tr >= 2:
            try:
                k = min(min_tr - 1, 3)
                Xtr, ytr = SMOTE(k_neighbors=k,
                                  random_state=42).fit_resample(Xtr, ytr)
            except Exception:
                pass

        m_b = CatBoostClassifier(iterations=200, depth=4,
                                  learning_rate=0.05, verbose=0,
                                  random_seed=i)
        m_b.fit(Xtr, ytr)
        yp = m_b.predict(Xte).flatten()
        fold_acc.append(accuracy_score(yte, yp))
        fold_f1.append(f1_score(yte, yp, average="weighted", zero_division=0))

    if fold_acc:
        bootstrap_acc.append(np.mean(fold_acc))
        bootstrap_f1.append(np.mean(fold_f1))

    if (i + 1) % 10 == 0:
        print(f"  Bootstrap {i+1}/{N_BOOTSTRAP}: "
              f"acc_avg={np.mean(bootstrap_acc):.3f}")

bootstrap_acc = np.array(bootstrap_acc)
bootstrap_f1 = np.array(bootstrap_f1)

print(f"\n=== Bootstrap Results (n={len(bootstrap_acc)} valid runs) ===")
print(f"\nAccuracy:")
print(f"  Mean:    {bootstrap_acc.mean():.3f}")
print(f"  Std:     {bootstrap_acc.std():.3f}")
print(f"  95% CI:  [{np.percentile(bootstrap_acc, 2.5):.3f}, "
      f"{np.percentile(bootstrap_acc, 97.5):.3f}]")

print(f"\nF1 Score:")
print(f"  Mean:    {bootstrap_f1.mean():.3f}")
print(f"  Std:     {bootstrap_f1.std():.3f}")
print(f"  95% CI:  [{np.percentile(bootstrap_f1, 2.5):.3f}, "
      f"{np.percentile(bootstrap_f1, 97.5):.3f}]")

# PART 3: Final model on all data
print("\n" + "=" * 70)
print("PART 3: Final Production Model")
print("=" * 70)

bc_full = np.bincount(y)
min_full = int(bc_full[bc_full > 0].min())
if min_full >= 2:
    k = min(min_full - 1, 5)
    X_final, y_final = SMOTE(k_neighbors=k,
                              random_state=42).fit_resample(X, y)
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

print("\nTop 10 features:")
print(importance.head(10).to_string(index=False))

# Save outputs
results_df.to_csv("features/L4_cv_results.csv", index=False)
importance.to_csv("features/L4_feature_importance.csv", index=False)
final_model.save_model("models/catboost_regime_classifier.cbm")

pd.DataFrame({
    "metric": ["accuracy", "f1"],
    "mean": [bootstrap_acc.mean(), bootstrap_f1.mean()],
    "std": [bootstrap_acc.std(), bootstrap_f1.std()],
    "ci_lower": [np.percentile(bootstrap_acc, 2.5),
                 np.percentile(bootstrap_f1, 2.5)],
    "ci_upper": [np.percentile(bootstrap_acc, 97.5),
                 np.percentile(bootstrap_f1, 97.5)]
}).to_csv("features/L4_bootstrap_ci.csv", index=False)

# Save predictions for plotting
df_pred = pd.read_csv("features/L3_zombie_features.csv")
if df_pred["regime"].nunique() == 1:
    df_pred.loc[df_pred["zombie_confirmed"] == 1, "regime"] = 3
    df_pred.loc[(df_pred["anomaly_score"] == -1) & (df_pred["regime"] == 0), "regime"] = 2
    df_pred.loc[(df_pred["mod_req_rate"] > 0) & (df_pred["regime"] == 0), "regime"] = 1
df_pred["predicted_regime"] = final_model.predict(
    df_pred[available].fillna(0).values).flatten()
df_pred.to_csv("features/L4_final_dataset.csv", index=False)

print("\n=== Saved files ===")
print("  features/L4_cv_results.csv")
print("  features/L4_feature_importance.csv")
print("  features/L4_bootstrap_ci.csv")
print("  features/L4_final_dataset.csv")
print("  models/catboost_regime_classifier.cbm")
print("\nDone.")
