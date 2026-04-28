"""
Convert this Python file to notebook with: jupytext --to ipynb research_pipeline_final.py
Or just paste cells one by one into a new notebook.
Each '# %%' marks a notebook cell.
"""

# %% [markdown]
# # E2E 5G SA Proactive Observability Framework
# ## Complete Research Pipeline — All 4 Layers
#
# **Testbed:** free5GC v4.2.1 + UERANSIM v3.2.8
# **Methodology:** RAN precursors → 5QI weighting → Zombie reconciliation → CatBoost regime classification

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import joblib
from pathlib import Path

os.chdir(Path.home() / 'obs_framework')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 11

print('Working dir:', os.getcwd())
print('Files:')
for d in ['data', 'features', 'models', 'plots']:
    if Path(d).exists():
        for f in sorted(Path(d).iterdir()):
            print(f'  {d}/{f.name}')

# %% [markdown]
# ## Layer 1 — RAN Precursor Signal

# %%
L1 = pd.read_csv('features/L1_ran_features.csv')
L1['t_rel'] = L1['time_window'] - L1['time_window'].min()

print(f'Shape: {L1.shape}')
print(f'Total ModReqs: {L1["mod_req_count"].sum()}')
print(f'Max ModReq rate: {L1["mod_req_rate"].max():.3f}/s')
print(f'RAN precursor events: {L1["ran_precursor_label"].sum()}')

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})

axes[0].plot(L1['t_rel'], L1['mod_req_rate'],
             color='#2980b9', lw=2, marker='o', markersize=5)
axes[0].fill_between(L1['t_rel'], L1['mod_req_rate'],
                     alpha=0.3, color='#2980b9')
axes[0].set_title('Layer 1 — PFCP Modification Request Rate (RAN Precursor Signal)',
                  fontweight='bold', fontsize=12)
axes[0].set_ylabel('ModReq/s')
axes[0].grid(alpha=0.3)

axes[1].bar(L1['t_rel'], L1['ran_precursor_label'],
            color='#c0392b', alpha=0.8, width=8)
axes[1].set_title('Detected RAN Precursor Events', fontweight='bold')
axes[1].set_xlabel('Time (s from capture start)')
axes[1].set_ylabel('Flag')
axes[1].set_ylim(0, 1.2)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/fig1_layer1_ran_precursor.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Layer 2 — 5QI Weighted Criticality

# %%
L2 = pd.read_csv('features/L2_5qi_features.csv')
L2['t_rel'] = L2['time_window'] - L2['time_window'].min()

print(f'Shape: {L2.shape}')
print(f'Avg GBR ratio: {L2["gbr_ratio"].mean():.2%}')
print(f'Avg voice ratio: {L2["voice_ratio"].mean():.2%}')

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(L2['t_rel'], L2['weighted_criticality'],
             color='#e67e22', lw=2)
axes[0].fill_between(L2['t_rel'], L2['weighted_criticality'],
                     alpha=0.3, color='#e67e22')
axes[0].plot(L2['t_rel'], L2['gbr_ratio']*100,
             '--', color='#c0392b', lw=1.5, label='GBR % (×100)')
axes[0].set_title('Layer 2 — 5QI Weighted Criticality + GBR Ratio',
                  fontweight='bold')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Score')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 5QI distribution from raw counts
qi_data = pd.Series({
    '5QI=1 Voice (GBR)':   L2['voice_pkts'].sum(),
    '5QI=8 BE-High':        L2['be_pkts'].sum() if 'be_pkts' in L2 else 0,
    '5QI=70 IoT':           L2.get('iot_pkts', pd.Series([0])).sum(),
    '5QI=9 BE':             max(L2['total_pkts'].sum() -
                               L2['voice_pkts'].sum() -
                               L2.get('be_pkts', pd.Series([0])).sum(), 0),
})
qi_data = qi_data[qi_data > 0]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6'][:len(qi_data)]
axes[1].pie(qi_data.values, labels=qi_data.index, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
axes[1].set_title('5QI Traffic Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/fig2_layer2_5qi.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Layer 3 — Zombie Session Detection
# **Key contribution:** SMF-UPF reconciliation eliminates false positives from anomaly detection alone

# %%
L3 = pd.read_csv('features/L3_zombie_features.csv')
L3['t_rel'] = L3['time_window'] - L3['time_window'].min()

n_anom = (L3['anomaly_score'] == -1).sum()
n_zomb = L3['zombie_confirmed'].sum()
n_fp = n_anom - n_zomb

print(f'Isolation Forest anomalies: {n_anom}')
print(f'Confirmed zombies after reconciliation: {n_zomb}')
print(f'False positives eliminated: {n_fp}')
print(f'FP reduction rate: {n_fp / max(n_anom, 1):.1%}')

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(L3['t_rel'], L3['session_delta'],
             color='#c0392b', lw=2, label='UPF-SMF Session Delta')
axes[0].plot(L3['t_rel'], L3['gtpu_silence_ratio']*5,
             '--', color='gray', lw=1.5, label='GTP-U Silence ×5')
zm = L3[L3['zombie_confirmed'] == 1]
if len(zm) > 0:
    axes[0].scatter(zm['t_rel'], zm['session_delta'],
                    color='#8e44ad', s=200, marker='X', zorder=5,
                    edgecolor='black', linewidth=2,
                    label=f'{len(zm)} Confirmed Zombies')
axes[0].set_title('Layer 3 — SMF-UPF Reconciliation', fontweight='bold')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Session Delta')
axes[0].legend()
axes[0].grid(alpha=0.3)

normal = L3[L3['anomaly_score'] == 1]['anomaly_raw_score']
anomaly = L3[L3['anomaly_score'] == -1]['anomaly_raw_score']
axes[1].hist(normal, bins=20, alpha=0.7, color='#27ae60',
             label=f'Normal (n={len(normal)})', edgecolor='black')
axes[1].hist(anomaly, bins=20, alpha=0.7, color='#c0392b',
             label=f'Anomaly (n={len(anomaly)})', edgecolor='black')
axes[1].set_title(f'Isolation Forest Score Distribution\n'
                  f'False Positives Eliminated: {n_fp}',
                  fontweight='bold')
axes[1].set_xlabel('Anomaly Score')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/fig3_layer3_zombie.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Layer 4 — CatBoost Regime Classifier with Layer Attribution

# %%
importance = pd.read_csv('features/L4_feature_importance.csv')
cv = pd.read_csv('features/L4_cv_results.csv')
ci = pd.read_csv('features/L4_bootstrap_ci.csv')

def tag_layer(feat):
    L1f = ['mod_req_rate', 'mod_req_rate_delta', 'acceleration',
           'session_churn_rate', 'failure_ratio']
    L2f = ['gbr_ratio', 'voice_ratio', 'weighted_criticality',
           'avg_weight', 'throughput_bps']
    L3f = ['session_delta', 'gtpu_silence_ratio',
           'zombie_confirmed', 'anomaly_raw_score']
    if feat in L1f: return 'L1_RAN'
    if feat in L2f: return 'L2_5QI'
    if feat in L3f: return 'L3_Zombie'
    return 'Cross_Layer'

importance['layer'] = importance['feature'].apply(tag_layer)

print('=== Cross-Validation Results ===')
print(cv.to_string(index=False))
print(f'\nMean Accuracy: {cv["accuracy"].mean():.3f} ± {cv["accuracy"].std():.3f}')
print(f'Mean F1:       {cv["f1"].mean():.3f} ± {cv["f1"].std():.3f}')

print('\n=== Bootstrap 95% Confidence Intervals ===')
for _, row in ci.iterrows():
    print(f'  {row["metric"].capitalize():<10}: '
          f'{row["mean"]:.3f} [{row["ci_lower"]:.3f}, {row["ci_upper"]:.3f}]')

print('\n=== Layer Contribution ===')
layer_totals = importance.groupby('layer')['importance'].sum().sort_values(ascending=False)
for layer, val in layer_totals.items():
    pct = val / layer_totals.sum() * 100
    print(f'  {layer:<15}: {val:>6.2f}  ({pct:.1f}%)')

# %%
# Combined Layer 4 figure: feature importance + layer pie
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 0.05], wspace=0.4)

# Feature bars
ax1 = fig.add_subplot(gs[0])
layer_colors = {
    'L1_RAN':      '#2980b9',
    'L2_5QI':      '#e67e22',
    'L3_Zombie':   '#8e44ad',
    'Cross_Layer': '#7f8c8d'
}
imp_sorted = importance.sort_values('importance', ascending=True)
bar_colors = [layer_colors[l] for l in imp_sorted['layer']]
bars = ax1.barh(imp_sorted['feature'], imp_sorted['importance'],
                color=bar_colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, imp_sorted['importance']):
    if val > 0.5:
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}', va='center', fontsize=9)
ax1.set_xlabel('Importance Score')
ax1.set_title('CatBoost Feature Importance by Layer', fontweight='bold')
legend_elements = [Patch(facecolor=c, edgecolor='black', label=l)
                   for l, c in layer_colors.items()]
ax1.legend(handles=legend_elements, loc='lower right')
ax1.grid(True, alpha=0.3, axis='x')

# Pie
ax2 = fig.add_subplot(gs[1])
layer_pie = importance.groupby('layer')['importance'].sum()
ax2.pie(layer_pie.values,
        labels=[f'{l}\n{v/layer_pie.sum()*100:.1f}%'
                for l, v in zip(layer_pie.index, layer_pie.values)],
        colors=[layer_colors[l] for l in layer_pie.index],
        startangle=90, explode=[0.05]*len(layer_pie),
        textprops={'fontsize': 10})
ax2.set_title('Cumulative\nLayer Contribution', fontweight='bold')

plt.suptitle('Layer 4 — Multi-Layer Feature Attribution',
             fontsize=13, fontweight='bold')
plt.savefig('plots/fig4_layer4_importance.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Final Combined Dashboard — All 4 Layers

# %%
df = pd.read_csv('features/L4_final_dataset.csv')
df['t_rel'] = df['time_window'] - df['time_window'].min()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0,0].plot(df['t_rel'], df['mod_req_rate'],
               color='#2980b9', lw=2, marker='o', markersize=4)
axes[0,0].fill_between(df['t_rel'], df['mod_req_rate'],
                       alpha=0.3, color='#2980b9')
axes[0,0].set_title('Layer 1 — RAN Precursor (PFCP ModReq Rate)',
                    fontweight='bold')
axes[0,0].set_xlabel('Time (s)')
axes[0,0].set_ylabel('ModReq/s')
axes[0,0].grid(alpha=0.3)

axes[0,1].plot(df['t_rel'], df['weighted_criticality'],
               color='#e67e22', lw=2)
axes[0,1].fill_between(df['t_rel'], df['weighted_criticality'],
                       alpha=0.3, color='#e67e22')
axes[0,1].set_title('Layer 2 — 5QI Weighted Criticality',
                    fontweight='bold')
axes[0,1].set_xlabel('Time (s)')
axes[0,1].set_ylabel('Criticality')
axes[0,1].grid(alpha=0.3)

axes[1,0].plot(df['t_rel'], df['session_delta'],
               color='#c0392b', lw=2, label='UPF-SMF Delta')
zm = df[df['zombie_confirmed'] == 1]
if len(zm) > 0:
    axes[1,0].scatter(zm['t_rel'], zm['session_delta'],
                      color='#8e44ad', s=180, marker='X', zorder=5,
                      edgecolor='black', linewidth=2,
                      label=f'{len(zm)} Zombies')
    axes[1,0].legend()
axes[1,0].set_title('Layer 3 — Zombie Detection (SMF-UPF Reconciliation)',
                    fontweight='bold')
axes[1,0].set_xlabel('Time (s)')
axes[1,0].set_ylabel('Session Delta')
axes[1,0].grid(alpha=0.3)

regime_colors_dict = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c', 3: '#8e44ad'}
regime_names = {0: 'Normal', 1: 'Low Stress',
                2: 'High Load', 3: 'Critical GBR'}
for r in sorted(df['predicted_regime'].unique()):
    mask = df['predicted_regime'] == r
    axes[1,1].scatter(df.loc[mask, 't_rel'],
                      df.loc[mask, 'predicted_regime'],
                      c=regime_colors_dict.get(r, 'gray'), s=80,
                      label=regime_names.get(r, f'R{r}'),
                      edgecolor='black', linewidth=0.5, alpha=0.8)
axes[1,1].set_title('Layer 4 — CatBoost Regime Classification',
                    fontweight='bold')
axes[1,1].set_xlabel('Time (s)')
axes[1,1].set_ylabel('Predicted Regime')
axes[1,1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
axes[1,1].set_yticks([0, 1, 2, 3])
axes[1,1].grid(alpha=0.3)

plt.suptitle('E2E 5G SA Proactive Observability Framework — All 4 Layers',
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('plots/fig5_combined_dashboard.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Final Statistical Summary (for paper)

# %%
print('=' * 70)
print('PAPER-READY RESULTS SUMMARY')
print('=' * 70)

print(f'\n[DATASET]')
print(f'  Time windows analyzed:      {len(df)}')
print(f'  PFCP messages captured:     {L1["total_msgs"].sum()}')
print(f'  GTP-U packets:              {L2["total_pkts"].sum() if "total_pkts" in L2 else "N/A"}')
print(f'  Feature dimensions:         {len(importance)}')

print(f'\n[LAYER 1 — RAN PRECURSOR]')
print(f'  Session Modifications:      {L1["mod_req_count"].sum()}')
print(f'  Max ModReq rate:            {L1["mod_req_rate"].max():.3f}/s')
print(f'  Precursor events flagged:   {L1["ran_precursor_label"].sum()}')

print(f'\n[LAYER 2 — 5QI CRITICALITY]')
print(f'  Avg weighted criticality:   {L2["weighted_criticality"].mean():.2f}')

print(f'\n[LAYER 3 — ZOMBIE DETECTION]')
print(f'  Anomalies (Isolation F):    {n_anom}')
print(f'  Confirmed zombies:          {n_zomb}')
print(f'  FP elimination rate:        {n_fp/max(n_anom,1):.1%}')

print(f'\n[LAYER 4 — REGIME CLASSIFICATION]')
print(f'  CV Accuracy:                {cv["accuracy"].mean():.3f} ± {cv["accuracy"].std():.3f}')
print(f'  CV F1:                      {cv["f1"].mean():.3f} ± {cv["f1"].std():.3f}')
acc_ci = ci[ci['metric']=='accuracy'].iloc[0]
f1_ci = ci[ci['metric']=='f1'].iloc[0]
print(f'  Bootstrap 95% CI Accuracy:  [{acc_ci["ci_lower"]:.3f}, {acc_ci["ci_upper"]:.3f}]')
print(f'  Bootstrap 95% CI F1:        [{f1_ci["ci_lower"]:.3f}, {f1_ci["ci_upper"]:.3f}]')

print(f'\n[LAYER CONTRIBUTION (feature importance %)]')
for layer, val in layer_totals.sort_values(ascending=False).items():
    print(f'  {layer:<15}: {val/layer_totals.sum()*100:>5.1f}%')

print(f'\n[OUTPUT FIGURES]')
for f in sorted(Path('plots').glob('fig*.png')):
    size = f.stat().st_size / 1024
    print(f'  {f} ({size:.0f} KB)')

print('\n' + '=' * 70)
