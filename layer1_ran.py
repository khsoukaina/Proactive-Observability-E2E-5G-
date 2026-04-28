import pandas as pd
import numpy as np

print("=== Layer 1: RAN Precursor Feature Extraction ===")

# Force all columns to string, parse types manually
pfcp = pd.read_csv("data/pfcp_events.csv",
                   names=["timestamp","msg_type","seid","cause","src","dst"],
                   dtype=str,
                   on_bad_lines='skip',
                   low_memory=False)
print(f"Raw rows loaded: {len(pfcp)}")

# Convert types manually with coercion
pfcp["timestamp"] = pd.to_numeric(pfcp["timestamp"], errors="coerce")
pfcp["msg_type"] = pd.to_numeric(pfcp["msg_type"], errors="coerce")

# Drop only rows with broken timestamp/msg_type
pfcp = pfcp.dropna(subset=["timestamp","msg_type"])
pfcp["msg_type"] = pfcp["msg_type"].astype(int)
print(f"Valid PFCP events: {len(pfcp)}")
print(f"\nMessage type distribution:")
print(pfcp["msg_type"].value_counts().head(15))

# 10-second windows
pfcp["time_window"] = (pfcp["timestamp"] // 10 * 10).astype(int)

window_features = pfcp.groupby("time_window").agg(
    total_msgs=("msg_type", "count"),
    mod_req_count=("msg_type", lambda x: (x == 52).sum()),
    mod_resp_count=("msg_type", lambda x: (x == 53).sum()),
    estab_count=("msg_type", lambda x: (x == 50).sum()),
    estab_resp_count=("msg_type", lambda x: (x == 51).sum()),
    del_count=("msg_type", lambda x: (x == 54).sum()),
    del_resp_count=("msg_type", lambda x: (x == 55).sum()),
    heartbeat_count=("msg_type", lambda x: x.isin([1,2]).sum()),
    other_count=("msg_type", lambda x: (~x.isin([1,2,50,51,52,53,54,55])).sum()),
).reset_index()

window_features["mod_req_rate"] = window_features["mod_req_count"] / 10.0
window_features["session_churn_rate"] = (
    (window_features["estab_count"] + window_features["del_count"]) /
    window_features["total_msgs"].clip(lower=1)
)
window_features["mod_req_rate_lag1"] = window_features["mod_req_rate"].shift(1).fillna(0)
window_features["mod_req_rate_delta"] = (
    window_features["mod_req_rate"] - window_features["mod_req_rate_lag1"]
)
window_features["acceleration"] = window_features["mod_req_rate_delta"].diff().fillna(0)
window_features["failure_ratio"] = 0.0
window_features["unique_sessions"] = window_features["estab_count"]

# Use total_msgs as proxy when ModReq is sparse
proxy_threshold = window_features["total_msgs"].quantile(0.75)
window_features["ran_precursor_label"] = (
    (window_features["mod_req_rate"] > 0) |
    (window_features["total_msgs"] > proxy_threshold)
).astype(int)

print(f"\n=== Layer 1 Summary ===")
print(f"Windows analyzed: {len(window_features)}")
print(f"Total PFCP messages: {window_features['total_msgs'].sum()}")
print(f"Total ModReqs: {window_features['mod_req_count'].sum()}")
print(f"Total Session Estabs: {window_features['estab_count'].sum()}")
print(f"Total Session Dels: {window_features['del_count'].sum()}")
print(f"Avg ModReq rate: {window_features['mod_req_rate'].mean():.4f}/s")
print(f"Max ModReq rate: {window_features['mod_req_rate'].max():.4f}/s")
print(f"RAN precursor events: {window_features['ran_precursor_label'].sum()}")

window_features.to_csv("features/L1_ran_features.csv", index=False)
print("\nSaved: features/L1_ran_features.csv")
