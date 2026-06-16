"""
Reproduit fidelement le style du diagramme original V5, mis a jour V6.
Sortie : architecture_v6.png  (150 dpi)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── utilitaires ────────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, fc, ec, lw=1.0, ls="-", radius=0.20, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle=ls, zorder=zorder)
    ax.add_patch(p)

def text(ax, x, y, s, fs=8, fw="normal", color="#222", ha="center", va="center",
         style="normal", zorder=5):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=color,
            ha=ha, va=va, style=style, zorder=zorder)

def arrow(ax, x1, y1, x2, y2, label="", col="#666", lw=1.2, fs=7.5,
          label_dx=0.07, label_dy=0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=11), zorder=6)
    if label:
        mx = (x1+x2)/2 + label_dx
        my = (y1+y2)/2 + label_dy
        ax.text(mx, my, label, fontsize=fs, color="#555",
                ha="left", va="center", style="italic", zorder=6)

def badge(ax, x, y, w, h, bg, txt, fs=8, ec="none"):
    rounded_box(ax, x, y, w, h, bg, ec, lw=0.8, zorder=5)
    text(ax, x+w/2, y+h/2, txt, fs=fs, fw="bold", color="white", zorder=6)

# ══════════════════════════════════════════════════════════════════════════
# TITRE
# ══════════════════════════════════════════════════════════════════════════
text(ax, 7, 8.82,
     "Architecture complète — système de maintenance prédictive V6",
     fs=11, fw="bold", color="#1a1a2e")

# ══════════════════════════════════════════════════════════════════════════
# ZONE IOT  (dashed vert, gauche)
# ══════════════════════════════════════════════════════════════════════════
rounded_box(ax, 0.15, 2.80, 2.55, 5.70, "none", "#2e7d32",
            lw=1.5, ls="--", radius=0.30, zorder=1)
text(ax, 0.35, 8.38, "Zone IoT terrain", fs=7.5, fw="bold",
     color="#2e7d32", ha="left")

# capteurs
sensor_labels = ["Capteur IFM 1", "Capteur IFM 2", "× 18 capteurs..."]
sensor_subs   = ["Temp + Vib X/Y/Z", "Temp + Vib X/Y/Z", "Temp + Vib X/Y/Z"]
for i, (lbl, sub) in enumerate(zip(sensor_labels, sensor_subs)):
    yy = 7.50 - i * 0.88
    # boite capteur
    rounded_box(ax, 0.25, yy-0.28, 2.35, 0.64, "white", "#2e7d32",
                lw=1.0, radius=0.15, zorder=3)
    # cercle capteur
    outer = plt.Circle((0.64, yy+0.04), 0.22, color="#c8e6c9",
                        ec="#2e7d32", lw=1.2, zorder=4)
    inner = plt.Circle((0.64, yy+0.04), 0.09, color="#2e7d32", zorder=5)
    ax.add_patch(outer); ax.add_patch(inner)
    text(ax, 1.60, yy+0.10, lbl, fs=7.5, fw="bold", color="#1b5e20")
    text(ax, 1.60, yy-0.10, sub, fs=6.5, color="#555")

# badge "20 capteurs IFM"
badge(ax, 0.42, 2.90, 1.98, 0.36, "#2e7d32", "20 capteurs IFM", fs=8)

# ══════════════════════════════════════════════════════════════════════════
# GATEWAY IOT
# ══════════════════════════════════════════════════════════════════════════
gw_x, gw_y, gw_w, gw_h = 3.10, 7.30, 2.90, 1.15
rounded_box(ax, gw_x, gw_y, gw_w, gw_h, "#dbeafe", "#3b82f6", lw=1.2, zorder=2)
text(ax, gw_x+gw_w/2, gw_y+0.82, "Gateway IoT", fs=9, fw="bold", color="#1e40af")
# puces
for j, line in enumerate(["Agrégation · 2 s / mesure", "Horodatage UTC"]):
    yy = gw_y + 0.52 - j*0.28
    circ = plt.Circle((gw_x+0.22, yy+0.05), 0.07, color="#3b82f6", zorder=5)
    ax.add_patch(circ)
    text(ax, gw_x+1.60, yy+0.04, line, fs=7.2, ha="center")

# fleches capteurs -> gateway
for yy in [7.54, 6.66, 5.78]:
    arrow(ax, 2.70, yy, 3.10, 7.72, col="#2e7d32", lw=0.9)

arrow(ax, gw_x+gw_w/2, gw_y, gw_x+gw_w/2, 6.82,
      label="export SQL", col="#3b82f6", label_dx=0.10)

# ══════════════════════════════════════════════════════════════════════════
# ai_cp.sql  (base de donnees)
# ══════════════════════════════════════════════════════════════════════════
db_x, db_y, db_w, db_h = 3.10, 5.65, 2.90, 1.12
rounded_box(ax, db_x, db_y, db_w, db_h, "#fef3c7", "#b45309", lw=1.2, zorder=2)
text(ax, db_x+db_w/2, db_y+0.80, "ai_cp.sql", fs=9, fw="bold", color="#92400e")
# icone cylindre (database)
for j, dh in enumerate([0.0, -0.07, -0.14]):
    e = Ellipse((db_x+0.42, db_y+0.45+dh), 0.34, 0.12,
                facecolor="#fef3c7", edgecolor="#b45309", lw=0.9, zorder=4)
    ax.add_patch(e)
ax.add_patch(FancyBboxPatch((db_x+0.25, db_y+0.29), 0.34, 0.17,
             boxstyle="square,pad=0", fc="#fef3c7", ec="#b45309", lw=0.9, zorder=3))
text(ax, db_x+1.65, db_y+0.50, "658 MB · MariaDB 10.4", fs=7.2, ha="center")
text(ax, db_x+1.65, db_y+0.28, "Export phpMyAdmin", fs=7.2, ha="center")

arrow(ax, db_x+db_w/2, db_y, db_x+db_w/2, 5.08,
      label="parsing Python", col="#b45309", label_dx=0.10)

# ══════════════════════════════════════════════════════════════════════════
# full_data.parquet  (1 seule boite centree, plus motor_mesure ni motors)
# ══════════════════════════════════════════════════════════════════════════
pq_x, pq_y, pq_w, pq_h = 3.60, 4.20, 2.40, 0.82
rounded_box(ax, pq_x, pq_y, pq_w, pq_h, "#f0fdf4", "#16a34a", lw=1.1, zorder=2)
# icone "page"
ax.add_patch(FancyBboxPatch((pq_x+0.18, pq_y+0.14), 0.34, 0.50,
             boxstyle="square,pad=0", fc="white", ec="#16a34a", lw=0.9, zorder=4))
for j in range(3):
    ax.plot([pq_x+0.24, pq_x+0.46], [pq_y+0.52-j*0.13, pq_y+0.52-j*0.13],
            color="#16a34a", lw=0.7, zorder=5)
text(ax, pq_x+1.40, pq_y+0.52, "full_data", fs=8, fw="bold", color="#15803d")
text(ax, pq_x+1.40, pq_y+0.28, "240 000 mes.", fs=7, color="#555")

# fleche parquet -> feature engineering
arrow(ax, pq_x+pq_w/2, pq_y, 4.60, 3.58, col="#555", label_dx=0.08)

# ══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (gauche du milieu)
# ══════════════════════════════════════════════════════════════════════════
fe_x, fe_y, fe_w, fe_h = 1.00, 2.10, 4.00, 1.45
rounded_box(ax, fe_x, fe_y, fe_w, fe_h, "#ede7f6", "#7b1fa2", lw=1.2, zorder=2)
text(ax, fe_x+fe_w/2, fe_y+1.14, "Feature engineering",
     fs=9, fw="bold", color="#4a148c")
text(ax, fe_x+2.30, fe_y+0.80, "31 features · fenêtre w = 20", fs=7.5, ha="center")
text(ax, fe_x+2.30, fe_y+0.56, "73 917 sessions · RobustScaler", fs=7.5, ha="center")
text(ax, fe_x+2.30, fe_y+0.32, "PCA 31 → 2  (95% variance)", fs=7.5, ha="center")
# icone soleil
cx, cy = fe_x+0.60, fe_y+0.62
circ3 = plt.Circle((cx, cy), 0.25, color="#ce93d8", ec="#7b1fa2", lw=1.0, zorder=4)
ax.add_patch(circ3)
for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
    dx, dy = np.cos(ang)*0.36, np.sin(ang)*0.36
    ax.plot([cx+np.cos(ang)*0.27, cx+dx],
            [cy+np.sin(ang)*0.27, cy+dy],
            color="#7b1fa2", lw=1.3, zorder=5)

arrow(ax, fe_x+fe_w, fe_y+fe_h/2, 5.20, fe_y+fe_h/2,
      label="pickle", col="#7b1fa2", label_dx=0.08)

# ══════════════════════════════════════════════════════════════════════════
# PIPELINE ML — 6 modeles  (droite du milieu)
# ══════════════════════════════════════════════════════════════════════════
pm_x, pm_y, pm_w, pm_h = 5.20, 2.10, 5.30, 1.45
rounded_box(ax, pm_x, pm_y, pm_w, pm_h, "#fce4ec", "#c62828", lw=1.2, zorder=2)
text(ax, pm_x+pm_w/2, pm_y+1.22, "Pipeline ML — 6 modèles",
     fs=9, fw="bold", color="#b71c1c")
text(ax, pm_x+pm_w/2, pm_y+0.98, "IF · LOF · OCSVM · ECOD · HBOS · COPOD",
     fs=7, color="#b71c1c", style="italic")

# 6 boites modeles
model_data = [
    ("IF",    "#ffcdd2", "#c62828"),
    ("LOF",   "#ffe0b2", "#e65100"),
    ("OCSVM", "#fff9c4", "#f57f17"),
    ("ECOD",  "#dcedc8", "#33691e"),
    ("HBOS",  "#b3e5fc", "#01579b"),
    ("COPOD", "#f8bbd0", "#880e4f"),
]
n = len(model_data)
bw, bh = 0.70, 0.46
gap = (pm_w - n*bw - 0.20) / (n-1)
for j, (name, fc, ec) in enumerate(model_data):
    bx = pm_x + 0.10 + j*(bw+gap)
    by = pm_y + 0.14
    rounded_box(ax, bx, by, bw, bh, fc, ec, lw=1.0, radius=0.10, zorder=3)
    text(ax, bx+bw/2, by+bh/2, name, fs=7.5, fw="bold")

# ══════════════════════════════════════════════════════════════════════════
# SoftVote badge (sous pipeline ML)
# ══════════════════════════════════════════════════════════════════════════
badge(ax, 5.25, 1.72, 5.20, 0.33, "#7b1fa2",
      "SoftVote RAPPORT : IF + ECOD + HBOS + COPOD", fs=7.5)

# fleches feature eng / pipeline -> api et dashboard
arrow(ax, fe_x+fe_w/2, fe_y, fe_x+fe_w/2+0.30, 1.40, col="#555")
arrow(ax, pm_x+pm_w/2, pm_y, pm_x+pm_w/2-0.30, 1.40, col="#555")

# ══════════════════════════════════════════════════════════════════════════
# API FastAPI V3.1.0
# ══════════════════════════════════════════════════════════════════════════
api_x, api_y, api_w, api_h = 1.00, 0.40, 3.60, 0.95
rounded_box(ax, api_x, api_y, api_w, api_h, "#dcfce7", "#16a34a", lw=1.2, zorder=2)
text(ax, api_x+api_w/2, api_y+0.73, "API FastAPI V3.1.0",
     fs=9, fw="bold", color="#15803d")
text(ax, api_x+2.10, api_y+0.46, "11 endpoints · port 8000", fs=7.2, ha="center")
text(ax, api_x+2.10, api_y+0.22, "Reponse ≤ 50 ms · 20 capteurs", fs=7.2, ha="center")
# icone accolades
text(ax, api_x+0.42, api_y+0.47, "{ }", fs=16, fw="bold", color="#15803d")

arrow(ax, api_x+api_w, api_y+api_h/2, 5.00, api_y+api_h/2,
      label="metrics", col="#16a34a", label_dx=0.08)

# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
dash_x, dash_y, dash_w, dash_h = 5.10, 0.40, 4.10, 0.95
rounded_box(ax, dash_x, dash_y, dash_w, dash_h, "#fef9c3", "#ca8a04", lw=1.2, zorder=2)
text(ax, dash_x+dash_w/2, dash_y+0.73, "Dashboard temps réel",
     fs=9, fw="bold", color="#92400e")
text(ax, dash_x+2.30, dash_y+0.46, "Feux · Jauge · Votes · Waveform", fs=7.2, ha="center")
text(ax, dash_x+2.30, dash_y+0.22, "Rafraîchissement 3 s · HTML5 + Chart.js", fs=7.2, ha="center")
# icone graphe barres
bx0 = dash_x + 0.22
for xi, hi, ci in [(bx0,      0.28, "#ca8a04"),
                    (bx0+0.18, 0.42, "#d97706"),
                    (bx0+0.36, 0.18, "#92400e")]:
    ax.add_patch(FancyBboxPatch((xi, dash_y+0.10), 0.14, hi,
                 boxstyle="square,pad=0", fc=ci, ec="none", zorder=4))

# ══════════════════════════════════════════════════════════════════════════
# LEGENDE
# ══════════════════════════════════════════════════════════════════════════
legend_items = [
    ("#c8e6c9", "#2e7d32", "Capteurs IFM"),
    ("#dbeafe", "#3b82f6", "Gateway IoT"),
    ("#fef3c7", "#b45309", "Source SQL"),
    ("#f0fdf4", "#16a34a", "Fichiers Parquet"),
    ("#ede7f6", "#7b1fa2", "Feature engineering"),
    ("#fce4ec", "#c62828", "Pipeline ML"),
    ("#dcfce7", "#16a34a", "API FastAPI"),
    ("#fef9c3", "#ca8a04", "Dashboard"),
]
lx = 0.20
ly = 0.02
for fc, ec, label in legend_items:
    r = FancyBboxPatch((lx, ly), 0.26, 0.24,
                       boxstyle="round,pad=0,rounding_size=0.05",
                       facecolor=fc, edgecolor=ec, lw=0.8, zorder=3)
    ax.add_patch(r)
    ax.text(lx+0.32, ly+0.12, label, fontsize=6.8, va="center",
            color="#333", zorder=4)
    lx += len(label)*0.085 + 0.50

plt.tight_layout(pad=0.1)
plt.savefig("architecture_v6.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("OK  architecture_v6.png genere")
