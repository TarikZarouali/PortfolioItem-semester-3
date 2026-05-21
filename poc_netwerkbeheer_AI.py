"""
Proof of Concept – AI vs Traditionele Monitoring in Netwerkbeheer
Auteur: Tarik Zarouali
Rapport: Wat kan AI betekenen voor het beheer van mijn netwerk?
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import json
import warnings
import webbrowser
import os
warnings.filterwarnings('ignore')

print("=" * 55)
print("  PoC – AI vs Traditionele Netwerkmonitoring")
print("=" * 55)

# ── Data inladen ──────────────────────────────
df = pd.read_csv('sampled_NF-CSE-CIC-IDS2018-v2.csv')
print(f"\n✅ Dataset geladen: {len(df):,} netwerkverbindingen")

# ── Features kiezen ───────────────────────────
features = [
    'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'SRC_TO_DST_AVG_THROUGHPUT',
    'DST_TO_SRC_AVG_THROUGHPUT', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
]
X      = df[features].fillna(0)
y_true = df['Label']

# ── Traditionele monitoring ───────────────────
drempel            = df['IN_BYTES'].mean() + 2 * df['IN_BYTES'].std()
traditioneel_alarm = (df['IN_BYTES'] > drempel).astype(int)
trad_gevonden      = int(((traditioneel_alarm == 1) & (y_true == 1)).sum())
trad_gemist        = int(((traditioneel_alarm == 0) & (y_true == 1)).sum())
trad_vals_alarm    = int(((traditioneel_alarm == 1) & (y_true == 0)).sum())

# ── AI detectie ───────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
model    = IsolationForest(contamination=0.12, random_state=42, n_estimators=100)
ai_pred  = model.fit_predict(X_scaled)
ai_alarm = (ai_pred == -1).astype(int)
ai_gevonden   = int(((ai_alarm == 1) & (y_true == 1)).sum())
ai_gemist     = int(((ai_alarm == 0) & (y_true == 1)).sum())
ai_vals_alarm = int(((ai_alarm == 1) & (y_true == 0)).sum())
werkelijke_aanvallen = int(y_true.sum())

print(f"   Traditioneel gevonden: {trad_gevonden:,} ({100*trad_gevonden/werkelijke_aanvallen:.1f}%)")
print(f"   AI gevonden          : {ai_gevonden:,} ({100*ai_gevonden/werkelijke_aanvallen:.1f}%)")

# ── Grafieken genereren ───────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Grafiek 1 – Staafdiagram
fig1, ax1 = plt.subplots(figsize=(7, 5), facecolor='#0d1117')
ax1.set_facecolor('#0d1117')
bars = ax1.bar(['Traditioneel\n(drempelwaarde)', 'AI\n(Isolation Forest)'],
               [trad_gevonden, ai_gevonden], color=['#f39c12', '#3498db'], width=0.45)
ax1.set_title('Aantal aanvallen gedetecteerd', fontweight='bold', color='white', pad=15)
ax1.set_ylabel('Gedetecteerde aanvallen', color='#aaaaaa')
ax1.set_ylim(0, werkelijke_aanvallen * 1.2)
ax1.axhline(y=werkelijke_aanvallen, color='#e74c3c', linestyle='--', alpha=0.7,
            label=f'Totaal aanvallen: {werkelijke_aanvallen:,}')
for bar, val in zip(bars, [trad_gevonden, ai_gevonden]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:,}\n({100*val/werkelijke_aanvallen:.1f}%)',
             ha='center', va='bottom', fontweight='bold', color='white', fontsize=11)
ax1.tick_params(colors='#aaaaaa')
ax1.spines[:].set_color('#333333')
ax1.legend(fontsize=9, labelcolor='white', facecolor='#1a1a2e', edgecolor='#333333')
ax1.grid(axis='y', alpha=0.15, color='white')
plt.tight_layout()
chart1 = fig_to_b64(fig1)
plt.close()

# Grafiek 2 – Per aanvalstype
attack_stats = df.groupby('Attack_Category').apply(
    lambda g: pd.Series({
        'Traditioneel': 100 * ((traditioneel_alarm[g.index] == 1) & (g['Label'] == 1)).sum() / max(g['Label'].sum(), 1),
        'AI':           100 * ((ai_alarm[g.index] == 1) & (g['Label'] == 1)).sum() / max(g['Label'].sum(), 1),
    })
).drop('Benign', errors='ignore')
fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor='#0d1117')
ax2.set_facecolor('#0d1117')
x = range(len(attack_stats))
ax2.bar([i - 0.175 for i in x], attack_stats['Traditioneel'], 0.35, label='Traditioneel', color='#f39c12', alpha=0.9)
ax2.bar([i + 0.175 for i in x], attack_stats['AI'],           0.35, label='AI',           color='#3498db', alpha=0.9)
ax2.set_title('Detectie per aanvalstype (%)', fontweight='bold', color='white', pad=15)
ax2.set_ylabel('Detectiepercentage (%)', color='#aaaaaa')
ax2.set_xticks(list(x))
ax2.set_xticklabels(attack_stats.index, rotation=25, ha='right', fontsize=9, color='#aaaaaa')
ax2.tick_params(colors='#aaaaaa')
ax2.spines[:].set_color('#333333')
ax2.legend(labelcolor='white', facecolor='#1a1a2e', edgecolor='#333333')
ax2.grid(axis='y', alpha=0.15, color='white')
ax2.set_ylim(0, 115)
plt.tight_layout()
chart2 = fig_to_b64(fig2)
plt.close()

# Grafiek 3 – Taartdiagram
fig3, ax3 = plt.subplots(figsize=(6, 5), facecolor='#0d1117')
ax3.set_facecolor('#0d1117')
cat_counts = df['Attack_Category'].value_counts()
pie_colors = ['#2ecc71', '#e74c3c', '#c0392b', '#922b21', '#7b241c', '#641e16', '#4a235a']
_, _, autotexts = ax3.pie(
    cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%',
    colors=pie_colors[:len(cat_counts)], startangle=140,
    textprops={'fontsize': 8, 'color': 'white'}
)
for at in autotexts:
    at.set_color('white')
ax3.set_title('Verdeling dataset', fontweight='bold', color='white', pad=15)
plt.tight_layout()
chart3 = fig_to_b64(fig3)
plt.close()

# ── Data naar JSON voor de HTML ───────────────
data = {
    "totaal":              len(df),
    "normaal":             int((df['Label'] == 0).sum()),
    "aanvallen":           werkelijke_aanvallen,
    "normaal_pct":         round(100 * (df['Label'] == 0).sum() / len(df), 1),
    "aanval_pct":          round(100 * werkelijke_aanvallen / len(df), 1),
    "trad_gevonden":       trad_gevonden,
    "trad_gemist":         trad_gemist,
    "trad_vals":           trad_vals_alarm,
    "trad_pct":            round(100 * trad_gevonden / werkelijke_aanvallen, 1),
    "ai_gevonden":         ai_gevonden,
    "ai_gemist":           ai_gemist,
    "ai_vals":             ai_vals_alarm,
    "ai_pct":              round(100 * ai_gevonden / werkelijke_aanvallen, 1),
    "verbetering":         ai_gevonden - trad_gevonden,
    "factor":              round(ai_gevonden / max(trad_gevonden, 1)),
    "chart1":              chart1,
    "chart2":              chart2,
    "chart3":              chart3,
}

base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, 'poc_data.json'), 'w') as f:
    json.dump(data, f)

print(f"\n✅ Data opgeslagen als: poc_data.json")
print("✅ Klaar!")
