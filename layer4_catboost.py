import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

print("=== Layer 4: CatBoost Regime Classifier ===")

df = pd.read_csv("features/L3_zombie_features.csv")

# Inject regime variation if all 0 (small dataset)
if df["regime"].nunique() == 1:
    print("Injecting regime variation from anomaly/zombie signals...")
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
X = df[available].fillna(0)
y = df["regime"].fillna(0).astype(int)

print(f"Features: {len(available)}, Samples: {len(X)}")
print(f"Regime distribution:\n{y.value_counts().sort_index()}")

split_idx = max(int(len(X) * 0.8), 1)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

if y_train.nunique() > 1:
    class_counts = y_train.value_counts()
    class_weights = {c: len(y_train)/(len(class_counts)*n)
                     for c, n in class_counts.items()}
    model = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=4,
        loss_function="MultiClass", class_weights=class_weights,
        random_seed=42, verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).flatten()
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    importance = pd.DataFrame({
        "feature": available,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False)
    print("\n=== Top Features ===")
    print(importance.head(10))

    importance.to_csv("features/L4_feature_importance.csv", index=False)
    model.save_model("models/catboost_regime_classifier.cbm")
    df["predicted_regime"] = model.predict(X).flatten()
    print("\nModel saved.")
else:
    df["predicted_regime"] = df["regime"]
    print("Single regime — using rule-based labels")

df.to_csv("features/L4_final_dataset.csv", index=False)
print("Saved: features/L4_final_dataset.csv")
