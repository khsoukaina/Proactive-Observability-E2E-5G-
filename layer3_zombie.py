import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

print("=== Layer 3: Zombie Session Detection ===")

df = pd.read_csv("features/L2_5qi_features.csv")
np.random.seed(42)
n = len(df)

# Synthesize SMF-UPF reconciliation signals
df["upf_session_count"] = df["unique_sessions"] + np.random.randint(0, 3, n)
df["smf_session_count"] = df["unique_sessions"]
df["session_delta"] = df["upf_session_count"] - df["smf_session_count"]

df["gtpu_silence_ratio"] = np.where(
    df["total_pkts"] > 0,
    1 - (df["total_pkts"] / max(df["total_pkts"].max(), 1)),
    1.0
)

df["session_age_norm"] = (
    df["time_window"] - df["time_window"].min()
) / max(1, df["time_window"].max() - df["time_window"].min())

zombie_features = [
    "session_delta", "gtpu_silence_ratio", "session_age_norm",
    "unique_sessions", "session_churn_rate", "failure_ratio", "mod_req_rate"
]

X = df[zombie_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(
    n_estimators=200, contamination=0.1, random_state=42
)
df["anomaly_score"] = iso_forest.fit_predict(X_scaled)
df["anomaly_raw_score"] = iso_forest.score_samples(X_scaled)

df["zombie_confirmed"] = (
    (df["anomaly_score"] == -1) &
    (df["session_delta"] > 0) &
    (df["gtpu_silence_ratio"] > 0.5)
).astype(int)

print(f"Windows analyzed: {len(df)}")
print(f"Isolation Forest anomalies: {(df['anomaly_score']==-1).sum()}")
print(f"Confirmed zombies (after FP filter): {df['zombie_confirmed'].sum()}")
print(f"FPs eliminated by reconciliation: "
      f"{(df['anomaly_score']==-1).sum() - df['zombie_confirmed'].sum()}")

joblib.dump(iso_forest, "models/isolation_forest_zombie.pkl")
joblib.dump(scaler, "models/zombie_scaler.pkl")
df.to_csv("features/L3_zombie_features.csv", index=False)
print("\nSaved: features/L3_zombie_features.csv")
print("Saved: models/isolation_forest_zombie.pkl")
