"""
train_ecod_only.py
==================
Génère UNIQUEMENT model_ecod_v3.pkl compatible avec les 25 features V5.
NE touche PAS aux autres modèles (IF, LOF, OCSVM, scaler, pca, features).

Usage :
    pip install pyod
    python train_ecod_only.py
"""

import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MODEL_DIR  = Path("models")
DATA_DIR   = Path("data")
DATA_FILE  = DATA_DIR / "dataset_2026_with_acc.csv"

print("=" * 60)
print("  ENTRAÎNEMENT ECOD UNIQUEMENT — Compatible V5 (25 features)")
print("=" * 60)

# ── 1. Vérifier que les fichiers V5 existent ─────────────────────────────────
required = ["scaler_v3.pkl", "pca_v3.pkl", "features_v3.pkl"]
for f in required:
    if not (MODEL_DIR / f).exists():
        print(f"❌ Fichier manquant : models/{f}")
        print("   Télécharge le ZIP PROJET_FINAL et copie le dossier models/")
        exit(1)

# ── 2. Charger le scaler et PCA existants (V5) ────────────────────────────────
print("\n[1/4] Chargement scaler et PCA V5...")
scaler        = joblib.load(MODEL_DIR / "scaler_v3.pkl")
pca           = joblib.load(MODEL_DIR / "pca_v3.pkl")
features_list = joblib.load(MODEL_DIR / "features_v3.pkl")

print(f"  ✅ Scaler : {scaler.n_features_in_} features")
print(f"  ✅ PCA    : {pca.n_components_} composantes")
print(f"  ✅ Features ({len(features_list)}) : {features_list}")

if scaler.n_features_in_ != 25:
    print(f"\n❌ Le scaler a {scaler.n_features_in_} features — pas 25.")
    print("   Tes modèles ont été écrasés par train_model_v3_unsupervised.py")
    print("   Retélécharge le dossier models/ depuis PROJET_FINAL.zip")
    exit(1)

# ── 3. Charger et préparer les données ───────────────────────────────────────
print(f"\n[2/4] Chargement données : {DATA_FILE}")
if not DATA_FILE.exists():
    print(f"❌ Fichier introuvable : {DATA_FILE}")
    exit(1)

df = pd.read_csv(DATA_FILE)
df = df.dropna()

# Calculer vib_total
df["vib_total"] = np.sqrt(
    df["vibration_x"]**2 + df["vibration_y"]**2 + df["vibration_z"]**2
)

# Construire les 25 features exactement comme extract_features() de l'API
def build_features(row):
    t   = row["temperature"]
    vx  = row["vibration_x"]
    vy  = row["vibration_y"]
    vz  = row["vibration_z"]
    vt  = row["vib_total"]
    cur = row.get("courant", row.get("current", 0.0)) or 0.0

    tn = max(0, min(1, (t - 25) / 40))
    vn = max(0, min(1, vt / 2598))
    cn = max(0, min(1, cur / 200))

    return {
        "temp_mean":    t,   "temp_std":    0.0, "temp_trend":  0.0, "temp_cur":    t,
        "vib_z_mean":   vz,  "vib_z_std":   0.0, "vib_z_rms_w": vz,
        "vib_z_kurt":   3.0, "vib_z_crest": 1.0, "vib_z_cur":   vz,
        "vib_x_mean":   vx,  "vib_x_std":   0.0, "vib_x_rms_w": vx, "vib_x_kurt":  3.0,
        "vib_y_mean":   vy,  "vib_y_std":   0.0, "vib_y_rms_w": vy, "vib_y_kurt":  3.0,
        "vib_total":    vt,
        "health_score": max(0, 100 * (1 - 0.30*tn - 0.30*vn - 0.20*cn - 0.20)),
        # Accélération
        "acc_p2p":      float(row.get("acc_p2p",   0) or 0),
        "acc_z2p":      float(row.get("acc_z2p",   0) or 0),
        "acc_crest":    float(row.get("acc_crest",  0) or 0),
        "acc_rms":      float(row.get("acc_rms",    0) or 0),
        # Courant
        "current_mean": float(cur),
    }

feat_df = df.apply(build_features, axis=1, result_type="expand")
# Réordonner selon l'ordre exact des features V5
feat_df = feat_df[features_list]
X = feat_df.values
print(f"  ✅ Dataset : {X.shape[0]:,} sessions × {X.shape[1]} features")

# ── 4. Appliquer le scaler et PCA existants ───────────────────────────────────
print("\n[3/4] Application scaler + PCA V5...")
X_sc  = scaler.transform(X)       # transform (pas fit_transform !)
X_pca = pca.transform(X_sc)       # transform (pas fit_transform !)
print(f"  ✅ X_pca shape : {X_pca.shape}")

# ── 5. Entraîner ECOD ─────────────────────────────────────────────────────────
print("\n[4/4] Entraînement ECOD...")
try:
    from pyod.models.ecod import ECOD
except ImportError:
    print("❌ pyod non installé — lance : pip install pyod")
    exit(1)

CONTAMINATION = 0.05
model_ecod = ECOD(contamination=CONTAMINATION)
model_ecod.fit(X_pca)

preds  = model_ecod.predict(X_pca)
n_ecod = (preds == 1).sum()
print(f"  ✅ ECOD | Anomalies : {n_ecod} ({n_ecod/len(X_pca)*100:.2f}%)")

# ── 6. Sauvegarde ─────────────────────────────────────────────────────────────
out_path = MODEL_DIR / "model_ecod_v3.pkl"
joblib.dump(model_ecod, out_path)
print(f"\n  ✅ Sauvegardé : {out_path} ({out_path.stat().st_size // 1024} KB)")

print("\n" + "=" * 60)
print("  ✅ ECOD V5 GÉNÉRÉ — Relance l'API :")
print("     python api_unified_pythagore.py")
print("  → Attendu : ✅ 4 modèles chargés | Features: 25 | PCA: 5")
print("=" * 60)
