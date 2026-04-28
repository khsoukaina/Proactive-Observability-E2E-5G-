import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

os.chdir(Path.home() / 'obs_framework')
print("Working dir:", os.getcwd())

# Reload all CSVs and force relative time conversion
L2 = pd.read_csv('features/L2_5qi_features.csv')
L3 = pd.read_csv('features/L3_zombie_features.csv')
df = pd.read_csv('features/L4_final_dataset.csv')

print(f"L2 shape: {L2.shape}")
print(f"L3 shape: {L3.shape}")
print(f"L4 shape: {df.shape}")

# Convert epoch to relative seconds
for d in [L2, L3, df]:
    d['t_rel'] = d['time_window'] - d['time_window'].min()

# === Fixed Layer 2 ===
print("\nGenerating Layer 2 plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(L2['t_rel'], L2['weighted_criticality'],
             color='#e67e22', lw=2, marker='s', markersize=4)
axes[0].fill_between(L2['t_rel'], L2['weighted_criticality'],
                     alpha=0.3, color='#e67e22')
ax2 = axes[0].twinx()
ax2.plot(L2['t_rel'], L2['gbr_ratio']*100,
         '--', color='#c0392b', lw=1.5, label='GBR % (x100)')
ax2.set_ylabel('GBR Ratio (%)', color='#c0392b')
ax2.tick_params(axis='y', labelcolor='#c0392b')
axes[0].set_title('Layer 2 - 5QI Weighted Criticality + GBR Ratio',
                  fontweight='bold')
axes[0].set_xlabel('Time (s from capture start)')
axes[0].set_ylabel('Criticality Score', color='#e67e22')
axes[0].tick_params(axis='y', labelcolor='#e67e22')
axes[0].grid(alpha=0.3)

# 5QI pie
qi = pd.Series({
    '5QI=1 Voice (GBR)': L2['voice_pkts'].sum(),
    '5QI=8 Best-Effort': max(L2['total_pkts'].sum() - L2['voice_pkts'].sum(), 0),
})
qi = qi[qi > 0]
axes[1].pie(qi.values, labels=qi.index,
            colors=['#e74c3c', '#3498db'],
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('5QI Traffic Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/fig2_layer2_5qi.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: plots/fig2_layer2_5qi.png")

# === Fixed Combined Dashboard ===
print("\nGenerating Combined Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# L1
axes[0,0].plot(df['t_rel'], df['mod_req_rate'],
               color='#2980b9', lw=2, marker='o', markersize=4)
axes[0,0].fill_between(df['t_rel'], df['mod_req_rate'],
                       alpha=0.3, color='#2980b9')
axes[0,0].set_title('Layer 1 - RAN Precursor (PFCP ModReq Rate)',
                    fontweight='bold')
axes[0,0].set_xlabel('Time (s)')
axes[0,0].set_ylabel('ModReq/s')
axes[0,0].grid(alpha=0.3)

# L2
axes[0,1].plot(df['t_rel'], df['weighted_criticality'],
               color='#e67e22', lw=2)
axes[0,1].fill_between(df['t_rel'], df['weighted_criticality'],
                       alpha=0.3, color='#e67e22')
axes[0,1].set_title('Layer 2 - 5QI Weighted Criticality',
                    fontweight='bold')
axes[0,1].set_xlabel('Time (s)')
axes[0,1].set_ylabel('Criticality')
axes[0,1].grid(alpha=0.3)

# L3
axes[1,0].plot(df['t_rel'], df['session_delta'],
               color='#c0392b', lw=2, label='UPF-SMF Delta')
zm = df[df['zombie_confirmed'] == 1]
if len(zm) > 0:
    axes[1,0].scatter(zm['t_rel'], zm['session_delta'],
                      color='#8e44ad', s=180, marker='X', zorder=5,
                      edgecolor='black', linewidth=2,
                      label=f'{len(zm)} Zombies')
    axes[1,0].legend()
axes[1,0].set_title('Layer 3 - Zombie Detection',
                    fontweight='bold')
axes[1,0].set_xlabel('Time (s)')
axes[1,0].set_ylabel('Session Delta')
axes[1,0].grid(alpha=0.3)

# L4
regime_colors = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c', 3: '#8e44ad'}
regime_names = {0: 'Normal', 1: 'Low Stress',
                2: 'High Load', 3: 'Critical GBR'}
for r in sorted(df['predicted_regime'].unique()):
    mask = df['predicted_regime'] == r
    axes[1,1].scatter(df.loc[mask, 't_rel'],
                      df.loc[mask, 'predicted_regime'],
                      c=regime_colors.get(r, 'gray'), s=80,
                      label=regime_names.get(r, f'R{r}'),
                      edgecolor='black', linewidth=0.5, alpha=0.8)
axes[1,1].set_title('Layer 4 - CatBoost Regime Classification',
                    fontweight='bold')
axes[1,1].set_xlabel('Time (s)')
axes[1,1].set_ylabel('Predicted Regime')
axes[1,1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
axes[1,1].set_yticks([0, 1, 2, 3])
axes[1,1].grid(alpha=0.3)

plt.suptitle('E2E 5G SA Proactive Observability Framework - All 4 Layers',
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('plots/fig5_combined_dashboard.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: plots/fig5_combined_dashboard.png")

print("\n=== Done ===")
print("\nFinal plot files:")
for p in sorted(Path('plots').glob('fig*.png')):
    size = p.stat().st_size / 1024
    print(f"  {p} ({size:.0f} KB)")
