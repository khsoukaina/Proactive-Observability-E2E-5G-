import pandas as pd
import numpy as np

print("=== Layer 2: 5QI-Aware Core Feature Extraction ===")

# 3GPP TS 23.501 — 5QI weight table for criticality scoring
QCI_WEIGHTS = {
    1:  {"weight": 1.0,  "type": "GBR",     "service": "Voice"},
    2:  {"weight": 0.95, "type": "GBR",     "service": "Video_Live"},
    3:  {"weight": 0.98, "type": "GBR",     "service": "Real_Time_Gaming"},
    4:  {"weight": 0.90, "type": "GBR",     "service": "Buffered_Video"},
    5:  {"weight": 0.85, "type": "Non_GBR", "service": "IMS_Signalling"},
    6:  {"weight": 0.70, "type": "Non_GBR", "service": "Video_TCP"},
    7:  {"weight": 0.75, "type": "Non_GBR", "service": "Voice_Video"},
    8:  {"weight": 0.50, "type": "Non_GBR", "service": "Best_Effort_High"},
    9:  {"weight": 0.20, "type": "Non_GBR", "service": "Best_Effort"},
    70: {"weight": 0.60, "type": "Non_GBR", "service": "IoT"},
    79: {"weight": 0.65, "type": "Non_GBR", "service": "V2X"},
    80: {"weight": 0.80, "type": "Non_GBR", "service": "Low_Latency"},
}

# Load Layer 1 features
L1 = pd.read_csv("features/L1_ran_features.csv")

# Load GTP-U flows
gtpu = pd.read_csv("data/gtpu_flows.csv",
                   names=["timestamp","src_ip","dst_ip","pkt_size","proto"],
                   on_bad_lines='skip', low_memory=False)
gtpu["timestamp"] = pd.to_numeric(gtpu["timestamp"], errors="coerce")
gtpu["pkt_size"] = pd.to_numeric(gtpu["pkt_size"], errors="coerce")
gtpu["proto"] = pd.to_numeric(gtpu["proto"], errors="coerce")
gtpu = gtpu.dropna(subset=["timestamp","pkt_size"])
print(f"GTP-U flows loaded: {len(gtpu)}")

# Heuristic 5QI classification by packet size + protocol
# This implements your paper's contribution: 5QI tagging at flow level
def classify_5qi(row):
    size = row["pkt_size"]
    proto = row["proto"] if not pd.isna(row["proto"]) else 17
    if proto == 17 and 100 <= size <= 300:
        return 1   # Voice (5QI=1, GBR critical)
    elif proto == 17 and size < 100:
        return 70  # IoT
    elif proto == 6 and size > 1000:
        return 9   # Best-effort download
    elif proto == 6 and size > 500:
        return 6   # Video/streaming
    elif proto == 17 and size > 500:
        return 4   # Buffered video
    else:
        return 8   # Default best-effort

gtpu["5qi"] = gtpu.apply(classify_5qi, axis=1)
gtpu["time_window"] = (gtpu["timestamp"] // 10 * 10).astype(int)
gtpu["5qi_weight"] = gtpu["5qi"].map(
    {k: v["weight"] for k, v in QCI_WEIGHTS.items()}
).fillna(0.3)

# Aggregate per 10s window
window_5qi = gtpu.groupby("time_window").agg(
    total_pkts=("pkt_size", "count"),
    total_bytes=("pkt_size", "sum"),
    avg_pkt_size=("pkt_size", "mean"),
    voice_pkts=("5qi", lambda x: (x == 1).sum()),
    be_pkts=("5qi", lambda x: (x == 9).sum()),
    iot_pkts=("5qi", lambda x: (x == 70).sum()),
    gbr_pkts=("5qi", lambda x: x.isin([1,2,3,4]).sum()),
    avg_weight=("5qi_weight", "mean"),
    weighted_criticality=("5qi_weight", lambda x: x.sum())
).reset_index()

window_5qi["gbr_ratio"] = (
    window_5qi["gbr_pkts"] / window_5qi["total_pkts"].clip(lower=1)
)
window_5qi["voice_ratio"] = (
    window_5qi["voice_pkts"] / window_5qi["total_pkts"].clip(lower=1)
)
window_5qi["throughput_bps"] = window_5qi["total_bytes"] * 8 / 10.0

# Merge with L1
combined = pd.merge(L1, window_5qi, on="time_window", how="outer").fillna(0)

# Regime classification (criticality-aware)
def classify_regime(row):
    if row["failure_ratio"] < 0.01 and row["gbr_ratio"] < 0.1:
        return 0  # Normal
    elif row["mod_req_rate"] < 0.05 and row["gbr_ratio"] < 0.3:
        return 1  # Low stress
    elif row["mod_req_rate"] >= 0.05 and row["gbr_ratio"] < 0.3:
        return 2  # High load, non-critical
    else:
        return 3  # Critical: GBR traffic at risk

combined["regime"] = combined.apply(classify_regime, axis=1)

print(f"\n=== Layer 2 Summary ===")
print(f"Windows: {len(combined)}")
print(f"5QI distribution in traffic:")
print(gtpu["5qi"].value_counts())
print(f"\nRegime distribution:")
print(combined["regime"].value_counts().sort_index())
print(f"Avg GBR ratio: {combined['gbr_ratio'].mean():.2%}")
print(f"Avg voice ratio: {combined['voice_ratio'].mean():.2%}")

combined.to_csv("features/L2_5qi_features.csv", index=False)
print("\nSaved: features/L2_5qi_features.csv")
