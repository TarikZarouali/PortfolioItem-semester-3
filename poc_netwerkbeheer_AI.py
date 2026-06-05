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
import os

warnings.filterwarnings('ignore')

print("PoC gestart - AI vs traditionele netwerkmonitoring")

# dataset inladen. elke rij is een netwerkstroom met een label (1 = aanval, 0 = normaal)
df = pd.read_csv('sampled_NF-CSE-CIC-IDS2018-v2.csv')
print(f"Dataset geladen: {len(df)} rijen")

# kolommen die ik gebruik als input voor het model
features = [
    'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'SRC_TO_DST_AVG_THROUGHPUT',
    'DST_TO_SRC_AVG_THROUGHPUT', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT'
]

# lege waarden vullen met 0 zodat het model niet crasht
X = df[features].fillna(0)  

# de echte labels om beide methodes mee te controleren
y_true = df['Label']        

# traditionele methode: simpele drempelwaarde op IN_BYTES
# alles boven gemiddelde + 2x standaarddeviatie zie ik als verdacht
gemiddelde = df['IN_BYTES'].mean()
std = df['IN_BYTES'].std()
drempel = gemiddelde + 2 * std

traditioneel_alarm = (df['IN_BYTES'] > drempel).astype(int)

# vergelijken met de echte labels: terecht gevonden, gemist, en vals alarm
trad_gevonden = int(((traditioneel_alarm == 1) & (y_true == 1)).sum())
trad_gemist = int(((traditioneel_alarm == 0) & (y_true == 1)).sum())
trad_vals_alarm = int(((traditioneel_alarm == 1) & (y_true == 0)).sum())

# AI methode: Isolation Forest
# eerst schalen, anders wegen grote getallen (bytes) veel zwaarder dan kleine
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# contamination = geschat aandeel afwijkend verkeer (~12%), random_state voor reproduceerbaarheid
model = IsolationForest(contamination=0.12, random_state=42, n_estimators=100)
ai_pred = model.fit_predict(X_scaled)
# de echte labels om beide methodes mee te controleren
ai_alarm = (ai_pred == -1).astype(int)  

ai_gevonden = int(((ai_alarm == 1) & (y_true == 1)).sum())
ai_gemist = int(((ai_alarm == 0) & (y_true == 1)).sum())
ai_vals_alarm = int(((ai_alarm == 1) & (y_true == 0)).sum())

werkelijke_aanvallen = int(y_true.sum())

print(f"Traditioneel gevonden: {trad_gevonden} van de {werkelijke_aanvallen}")
print(f"AI gevonden: {ai_gevonden} van de {werkelijke_aanvallen}")


# helper functie om grafiek naar base64 te zetten zodat ik het in de html kan zetten
# (scheelt losse png-bestanden op schijf)
def grafiek_naar_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return b64


# grafiek 1: staafdiagram dat beide methodes vergelijkt
fig1, ax1 = plt.subplots(figsize=(7, 5), facecolor='#0d1117')
ax1.set_facecolor('#0d1117')

labels = ['Traditioneel\n(drempelwaarde)', 'AI\n(Isolation Forest)']
waarden = [trad_gevonden, ai_gevonden]
kleuren = ['#f39c12', '#3498db']

bars = ax1.bar(labels, waarden, color=kleuren, width=0.45)

ax1.set_title('Aantal aanvallen gedetecteerd', fontweight='bold', color='white', pad=15)
ax1.set_ylabel('Gedetecteerde aanvallen', color='#aaaaaa')
ax1.set_ylim(0, werkelijke_aanvallen * 1.2)
# rode stippellijn = totaal aantal aanvallen, zo zie je meteen hoeveel er gemist wordt
ax1.axhline(y=werkelijke_aanvallen, color='#e74c3c', linestyle='--', alpha=0.7,
            label=f'Totaal aanvallen: {werkelijke_aanvallen}')

# aantal + percentage boven elke balk zetten
for bar, val in zip(bars, waarden):
    pct = round(100 * val / werkelijke_aanvallen, 1)
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             f'{val}\n({pct}%)', ha='center', va='bottom',
             fontweight='bold', color='white', fontsize=11)

ax1.tick_params(colors='#aaaaaa')
ax1.spines[:].set_color('#333333')
ax1.legend(fontsize=9, labelcolor='white', facecolor='#1a1a2e', edgecolor='#333333')
ax1.grid(axis='y', alpha=0.15, color='white')
plt.tight_layout()
chart1 = grafiek_naar_base64(fig1)
plt.close()

# grafiek 2: per aanvalstype kijken hoe goed beide methodes zijn
categorieen = df['Attack_Category'].unique()
trad_pcts = []
ai_pcts = []
cat_namen = []

for cat in categorieen:
    if cat == 'Benign':  # normaal verkeer, geen aanval, dus overslaan
        continue
    mask = df['Attack_Category'] == cat
    totaal_cat = int(df.loc[mask, 'Label'].sum())
    if totaal_cat == 0:  # geen aanvallen in deze categorie, niets te meten
        continue
    # percentage dat elke methode binnen deze categorie oppakt
    trad_pct_cat = 100 * int(((traditioneel_alarm[mask] == 1) & (df.loc[mask, 'Label'] == 1)).sum()) / totaal_cat
    ai_pct_cat = 100 * int(((ai_alarm[mask] == 1) & (df.loc[mask, 'Label'] == 1)).sum()) / totaal_cat
    cat_namen.append(cat)
    trad_pcts.append(trad_pct_cat)
    ai_pcts.append(ai_pct_cat)

fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor='#0d1117')
ax2.set_facecolor('#0d1117')

# twee balken naast elkaar per categorie
x = list(range(len(cat_namen)))
ax2.bar([i - 0.175 for i in x], trad_pcts, 0.35, label='Traditioneel', color='#f39c12', alpha=0.9)
ax2.bar([i + 0.175 for i in x], ai_pcts, 0.35, label='AI', color='#3498db', alpha=0.9)

ax2.set_title('Detectie per aanvalstype (%)', fontweight='bold', color='white', pad=15)
ax2.set_ylabel('Detectiepercentage (%)', color='#aaaaaa')
ax2.set_xticks(x)
ax2.set_xticklabels(cat_namen, rotation=25, ha='right', fontsize=9, color='#aaaaaa')
ax2.tick_params(colors='#aaaaaa')
ax2.spines[:].set_color('#333333')
ax2.legend(labelcolor='white', facecolor='#1a1a2e', edgecolor='#333333')
ax2.grid(axis='y', alpha=0.15, color='white')
ax2.set_ylim(0, 115)
plt.tight_layout()
chart2 = grafiek_naar_base64(fig2)
plt.close()

# grafiek 3: verdeling van de dataset (taartdiagram)
fig3, ax3 = plt.subplots(figsize=(6, 5), facecolor='#0d1117')
ax3.set_facecolor('#0d1117')

cat_counts = df['Attack_Category'].value_counts()
kleuren_taart = ['#2ecc71', '#e74c3c', '#c0392b', '#922b21', '#7b241c', '#641e16', '#4a235a']

_, _, autotexts = ax3.pie(
    cat_counts.values,
    labels=cat_counts.index,
    autopct='%1.1f%%',
    colors=kleuren_taart[:len(cat_counts)],
    startangle=140,
    textprops={'fontsize': 8, 'color': 'white'}
)

for t in autotexts:
    t.set_color('white')  # percentages in de taart wit maken zodat ze leesbaar zijn

ax3.set_title('Verdeling dataset', fontweight='bold', color='white', pad=15)
plt.tight_layout()
chart3 = grafiek_naar_base64(fig3)
plt.close()

# alles samenvoegen in een dict en opslaan als json
normaal_count = int((df['Label'] == 0).sum())

data = {
    "totaal": len(df),
    "normaal": normaal_count,
    "aanvallen": werkelijke_aanvallen,
    "normaal_pct": round(100 * normaal_count / len(df), 1),
    "aanval_pct": round(100 * werkelijke_aanvallen / len(df), 1),
    "trad_gevonden": trad_gevonden,
    "trad_gemist": trad_gemist,
    "trad_vals": trad_vals_alarm,
    "trad_pct": round(100 * trad_gevonden / werkelijke_aanvallen, 1),
    "ai_gevonden": ai_gevonden,
    "ai_gemist": ai_gemist,
    "ai_vals": ai_vals_alarm,
    "ai_pct": round(100 * ai_gevonden / werkelijke_aanvallen, 1),
    "verbetering": ai_gevonden - trad_gevonden,
    "factor": round(ai_gevonden / max(trad_gevonden, 1)),  # max(...,1) voorkomt delen door 0
    "chart1": chart1,
    "chart2": chart2,
    "chart3": chart3
}

# json wegschrijven in dezelfde map als dit script, ongeacht vanwaar ik het start
base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, 'poc_data.json'), 'w') as f:
    json.dump(data, f)

print("Klaar! Data opgeslagen in poc_data.json")