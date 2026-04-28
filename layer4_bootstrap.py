import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

print("=== Bootstrap Confidence Intervals on CV Performance ===")

df = pd.read_csv("features/L3_zombie_features.csv")

if df["regime"].nunique() == 1:
    df.loc[df["zombie_confirmed"] == 1, "regime"] = 3
    df.loc[(df["anomaly_score"] == -1) & (df["regime"] == 0), "regime"] = 2
    df.loc[(df["mod_req_rate"] > 0) & (df["regime"] == 0), "regime"] = 1

FEATURES = [f for f in [
    "mod_req_rate", "mod_req_rate_delta", "acceleration",
    "session_churn_rate", "failure_ratio",
    "gbr_ratio", "voice_ratio", "weighted_criticality",
    "avg_weight", "throughput_bps",
    "session_delta", "gtpu_silence_ratio",
    "zombie_confirmed", "anomaly_raw_score",
    "unique_sessions", "total_pkts",
] if f in df.columns]

X_all = df[FEATURES].fillna(0).values
y_all = df["regime"].fillna(0).astype(int).values

# Filter rare classes
counts = pd.Series(y_all).value_counts()
keep = counts[counts >= 2].index.tolist()
mask = np.isin(y_all, keep)
X_all, y_all = X_all[mask], y_all[mask]

N_BOOTSTRAP = 100
print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations with stratified CV...")

bootstrap_acc = []
bootstrap_f1 = []
np.random.seed(42)

for i in range(N_BOOTSTRAP):
    # Bootstrap resample
    idx = np.random.choice(len(X_all), size=len(X_all), replace=True)
    X_b, y_b = X_all[idx], y_all[idx]

    # Stratified CV on the bootstrap sample
    if len(np.unique(y_b)) < 2:
        continue
    min_class = np.bincount(y_b)[np.bincount(y_b) > 0].min()
    if min_class < 2:
        continue

    n_splits = min(3, min_class)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
    fold_acc, fold_f1 = [], []

    for tr, te in skf.split(X_b, y_b):
        Xtr, Xte = X_b[tr], X_b[te]
        ytr, yte = y_b[tr], y_b[te]

        # SMOTE
        tr_counts = np.bincount(ytr)
        min_tr = tr_counts[tr_counts > 0].min()
        if min_tr >= 2:
            k = min(min_tr - 1, 3)
            try:
                Xtr, ytr = SMOTE(k_neighbors=k, random_state=42).fit_resample(Xtr, ytr)
            except: pass

        model = CatBoostClassifier(iterations=200, depth=4,
                                    learning_rate=0.05, verbose=0,
                                    random_seed=i)
        model.fit(Xtr, ytr)
        yp = model.predict(Xte).flatten()
        fold_acc.append(accuracy_score(yte, yp))
        fold_f1.append(f1_score(yte, yp, average="weighted", zero_division=0))

    if fold_acc:
        bootstrap_acc.append(np.mean(fold_acc))
        bootstrap_f1.append(np.mean(fold_f1))

    if (i+1) % 20 == 0:
        print(f"  Iter {i+1}/{N_BOOTSTRAP}: "
              f"acc={np.mean(bootstrap_acc):.3f}, "
              f"f1={np.mean(bootstrap_f1):.3f}")

# 95% CI
bootstrap_acc = np.array(bootstrap_acc)
bootstrap_f1 = np.array(bootstrap_f1)

print(f"\n=== Bootstrap Results (n={len(bootstrap_acc)} valid iterations) ===")
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

# Save for paper
pd.DataFrame({
    "metric": ["accuracy", "f1"],
    "mean": [bootstrap_acc.mean(), bootstrap_f1.mean()],
    "std": [bootstrap_acc.std(), bootstrap_f1.std()],
    "ci_lower": [np.percentile(bootstrap_acc, 2.5),
                 np.percentile(bootstrap_f1, 2.5)],
    "ci_upper": [np.percentile(bootstrap_acc, 97.5),
                 np.percentile(bootstrap_f1, 97.5)]
}).to_csv("features/L4_bootstrap_ci.csv", index=False)

print(f"\nSaved: features/L4_bootstrap_ci.csv")
