"""
retrain_from_real_data.py
=========================
Réentraîne les 4 modèles ML sur les VRAIES données IFM du SQL.

Problème actuel :
  - Les modèles existants ont été entraînés sur un dataset avec
    vib_x_médiane=4mg et vib_y_médiane=3mg (valeurs très basses)
  - Les données réelles du SQL ont vib_x~152mg, vib_y~153mg
  - Résultat : score=1.0 pour tout → tous les capteurs = CRITIQUE

Ce script :
  1. Charge dataset_2026_with_acc.csv (vraies données IFM)
  2. Calcule les 25 features exactes de l'API (extract_features)
  3. Réentraîne RobustScaler + PCA + 4 modèles sur ces vraies distributions
  4. Sauvegarde dans models/ — compatible API sans modification

Usage :
  python retrain_from_real_data.py
  python retrain_from_real_data.py --contamination 0.05
  python retrain_from_real_data.py --csv data/dataset_2026_with_acc.csv
"""

import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kurtosis

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DEFAULT_CSV         = "data/dataset_real_full.csv"  # Fichier généré par generate_dataset_from_sql.py (clé corrigée)
# 606 106 mesures réelles vs 5 120 avec l'ancienne clé bugguée
MODEL_DIR           = Path("models")
WINDOW_SIZE         = 10      # même valeur que l'API
CONTAMINATION       = 0.05    # 5% supposés anormaux
RANDOM_STATE        = 42

# Seuils P95 calculés depuis les vraies données IFM
SEUIL_TEMP_MAX      = 50.7    # °C  — P95 réel
SEUIL_VIB_MAX       = 576.0   # mg  — P95 réel vib_z
SEUIL_COURANT       = 100.0   # A   — valeur industrielle

# Ordre exact des 25 features — IDENTIQUE à l'API
FEATURES_ORDER = [
    "temp_mean",   "temp_std",    "temp_trend",  "temp_cur",
    "vib_z_mean",  "vib_z_std",   "vib_z_rms_w", "vib_z_kurt",
    "vib_z_crest", "vib_z_cur",
    "vib_x_mean",  "vib_x_std",   "vib_x_rms_w", "vib_x_kurt",
    "vib_y_mean",  "vib_y_std",   "vib_y_rms_w", "vib_y_kurt",
    "vib_total",   "health_score",
    "acc_p2p",     "acc_z2p",     "acc_crest",   "acc_rms",
    "current_mean",
]

G  = "\033[92m"
R  = "\033[91m"
Y  = "\033[93m"
C  = "\033[96m"
B  = "\033[1m"
RS = "\033[0m"

def ok(msg):   print(f"  {G}✅ {msg}{RS}")
def err(msg):  print(f"  {R}❌ {msg}{RS}")
def info(msg): print(f"  {C}ℹ  {msg}{RS}")
def warn(msg): print(f"  {Y}⚠  {msg}{RS}")
def head(msg): print(f"\n{B}{C}{'═'*60}\n  {msg}\n{'═'*60}{RS}")


# ══════════════════════════════════════════════════════════════════
# FONCTIONS DE CALCUL DES FEATURES
# (identiques à extract_features() dans api_unified_pythagore.py)
# ══════════════════════════════════════════════════════════════════

def safe_rms(arr):
    """Root Mean Square — énergie du signal."""
    a = np.array(arr, dtype=float)
    return float(np.sqrt(np.mean(a ** 2))) if len(a) > 0 else 0.0

def safe_kurtosis(arr):
    """Kurtosis — impulsivité du signal. 3.0 = distribution normale."""
    a = np.array(arr, dtype=float)
    if len(a) < 4 or np.std(a) == 0:
        return 3.0
    try:
        return float(kurtosis(a, fisher=False))
    except Exception:
        return 3.0

def safe_crest(arr):
    """Crest factor = max / RMS. > 3 indique des chocs."""
    a = np.array(arr, dtype=float)
    r = safe_rms(a)
    return float(np.max(np.abs(a)) / r) if r > 0 else 1.0

def safe_trend(arr):
    """Pente de régression linéaire — tendance positive = dégradation."""
    a = np.array(arr, dtype=float)
    if len(a) < 2:
        return 0.0
    x = np.arange(len(a), dtype=float)
    try:
        return float(np.polyfit(x, a, 1)[0])
    except Exception:
        return 0.0

VIB_TOTAL_MAX = float(np.sqrt(3) * 1500)  # ≈ 2598 mg — identique à l API

def _norm01(x, lo, hi):
    """Normalisation 0-1 par plage — identique à l API."""
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))

def compute_health(temp_mean, vib_total_3d, vib_z_kurt=3.0, current_mean=0.0):
    """
    Score de santé 0-100 — IDENTIQUE à la formule de api_unified_pythagore.py.
    Poids : temp 30%, vib_total_3D 30%, kurtosis 25%, courant 15%.
    100 = moteur parfait | 0 = moteur critique.

    Paramètres:
        temp_mean    : température moyenne fenêtre (°C)
        vib_total_3d : norme 3D moyenne √(X²+Y²+Z²) sur la fenêtre (mg)
        vib_z_kurt   : kurtosis de vib_z (impulsivité — 3.0 = normal)
        current_mean : courant moyen (A) — 0 si non disponible
    """
    temp_n = _norm01(temp_mean,    25,  65)
    vib_n  = _norm01(vib_total_3d,  0,  VIB_TOTAL_MAX)
    kurt_n = _norm01(vib_z_kurt,    0,  10)
    cur_n  = _norm01(current_mean,  0, 200)
    score  = 100.0 * (1.0 - 0.30*temp_n - 0.30*vib_n - 0.25*kurt_n - 0.15*cur_n)
    return round(max(0.0, min(100.0, score)), 1)

def extract_window_features(window_df: pd.DataFrame) -> dict:
    """
    Calcule exactement les 25 features depuis une fenêtre de WINDOW_SIZE mesures.
    DOIT être identique à extract_features() dans api_unified_pythagore.py.
    """
    T   = window_df["temperature"].values.astype(float)
    VX  = window_df["vibration_x"].values.astype(float)
    VY  = window_df["vibration_y"].values.astype(float)
    VZ  = window_df["vibration_z"].values.astype(float)
    I   = window_df.get("current", pd.Series([0.0]*len(window_df))).values.astype(float) \
          if "current" in window_df.columns else np.zeros(len(window_df))

    # Norme vectorielle 3D
    VT  = np.sqrt(VX**2 + VY**2 + VZ**2)

    # Accélération estimée (dérivée de vib_z)
    acc = np.diff(VZ, prepend=VZ[0])

    return {
        # Température (4)
        "temp_mean":    float(np.mean(T)),
        "temp_std":     float(np.std(T)),
        "temp_trend":   safe_trend(T),
        "temp_cur":     float(T[-1]),
        # Vibration Z (6)
        "vib_z_mean":   float(np.mean(VZ)),
        "vib_z_std":    float(np.std(VZ)),
        "vib_z_rms_w":  safe_rms(VZ),
        "vib_z_kurt":   safe_kurtosis(VZ),
        "vib_z_crest":  safe_crest(VZ + 1e-9),
        "vib_z_cur":    float(VZ[-1]),
        # Vibration X (4)
        "vib_x_mean":   float(np.mean(VX)),
        "vib_x_std":    float(np.std(VX)),
        "vib_x_rms_w":  safe_rms(VX),
        "vib_x_kurt":   safe_kurtosis(VX),
        # Vibration Y (4)
        "vib_y_mean":   float(np.mean(VY)),
        "vib_y_std":    float(np.std(VY)),
        "vib_y_rms_w":  safe_rms(VY),
        "vib_y_kurt":   safe_kurtosis(VY),
        # Combinées (2)
        "vib_total":    float(np.mean(VT)),
        # health_score identique à l API : utilise vib_total_3D + kurtosis vib_z
        "health_score": compute_health(
            temp_mean    = float(np.mean(T)),
            vib_total_3d = float(np.mean(VT)),   # norme 3D √(X²+Y²+Z²)
            vib_z_kurt   = safe_kurtosis(VZ),     # impulsivité
            current_mean = float(np.mean(I)),
        ),
        # Accélération (4)
        "acc_p2p":      float(np.max(acc) - np.min(acc)),
        "acc_z2p":      float(np.max(np.abs(acc))),
        "acc_crest":    safe_crest(acc + 1e-9),
        "acc_rms":      safe_rms(acc),
        # Courant (1)
        "current_mean": float(np.mean(I)),
    }


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — CHARGER LE CSV
# ══════════════════════════════════════════════════════════════════

def load_dataset(csv_path: str) -> pd.DataFrame:
    head("ÉTAPE 1 — CHARGEMENT DU DATASET")

    p = Path(csv_path)
    if not p.exists():
        err(f"Fichier introuvable : {csv_path}")
        err("Lance d'abord : python generate_dataset_from_sql.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["temperature", "vibration_z"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    # Ajouter colonne current si absente
    if "current" not in df.columns:
        df["current"] = 0.0

    ok(f"Dataset : {len(df):,} mesures | {df['sensor_id'].nunique()} capteurs")
    info(f"Capteurs : {sorted(df['sensor_id'].unique())}")
    info(f"Période  : {df['timestamp'].min()} → {df['timestamp'].max()}")
    info(f"Temp     : {df['temperature'].min():.1f}°C → {df['temperature'].max():.1f}°C")
    info(f"Vib Z    : {df['vibration_z'].min():.1f} → {df['vibration_z'].max():.1f} mg")
    info(f"Vib X    : {df['vibration_x'].min():.1f} → {df['vibration_x'].max():.1f} mg")

    return df


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — CONSTRUIRE LA MATRICE DE FEATURES
# ══════════════════════════════════════════════════════════════════

def build_feature_matrix(df: pd.DataFrame):
    head("ÉTAPE 2 — CALCUL DES 25 FEATURES")
    info(f"Fenêtre glissante : {WINDOW_SIZE} mesures par capteur")

    all_features     = []
    heuristic_labels = []   # 1=anomalie heuristique pour évaluation seulement
    n_windows        = 0

    for sid in df["sensor_id"].unique():
        sdf = df[df["sensor_id"] == sid].reset_index(drop=True)

        if len(sdf) < WINDOW_SIZE:
            warn(f"Capteur {sid} : {len(sdf)} mesures < {WINDOW_SIZE} → ignoré")
            continue

        for i in range(len(sdf) - WINDOW_SIZE + 1):
            window = sdf.iloc[i : i + WINDOW_SIZE]
            try:
                feats = extract_window_features(window)
                row   = [feats[f] for f in FEATURES_ORDER]

                # Vérifier qu'il n'y a pas de NaN/inf
                if any(not np.isfinite(v) for v in row if v is not None):
                    continue

                all_features.append(row)
                n_windows += 1

                # Étiquette heuristique (pour stats, pas pour entraînement)
                is_anom = (
                    feats["temp_mean"]   > SEUIL_TEMP_MAX or
                    feats["vib_z_rms_w"] > SEUIL_VIB_MAX  or
                    feats["current_mean"] > SEUIL_COURANT
                )
                heuristic_labels.append(1 if is_anom else 0)

            except Exception as e:
                continue

    if not all_features:
        err("Aucune feature calculée — vérifie le dataset")
        sys.exit(1)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(heuristic_labels, dtype=int)

    # Remplacer NaN/inf résiduels
    X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)

    n_anom = int(y.sum())
    ok(f"Matrice : {X.shape[0]:,} sessions × {X.shape[1]} features")
    ok(f"Anomalies heuristiques : {n_anom} ({100*n_anom/len(y):.1f}%)")

    # Afficher les stats des features clés
    feat_idx = {f: i for i, f in enumerate(FEATURES_ORDER)}
    info(f"temp_mean  : moy={X[:, feat_idx['temp_mean']].mean():.1f}°C  "
         f"std={X[:, feat_idx['temp_mean']].std():.1f}")
    info(f"vib_z_mean : moy={X[:, feat_idx['vib_z_mean']].mean():.1f}mg  "
         f"std={X[:, feat_idx['vib_z_mean']].std():.1f}")
    info(f"vib_x_mean : moy={X[:, feat_idx['vib_x_mean']].mean():.1f}mg  "
         f"std={X[:, feat_idx['vib_x_mean']].std():.1f}")

    return X, y


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — SCALER + PCA
# ══════════════════════════════════════════════════════════════════

def preprocess(X: np.ndarray):
    head("ÉTAPE 3 — PRÉTRAITEMENT (RobustScaler + PCA)")

    # RobustScaler : médiane + IQR — résistant aux outliers industriels
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    ok(f"RobustScaler : {X_scaled.shape[1]} features normalisées")
    info(f"Médiane temp_mean apprise  : {scaler.center_[0]:.2f}°C")
    info(f"Médiane vib_z_mean apprise : {scaler.center_[4]:.2f}mg")
    info(f"Médiane vib_x_mean apprise : {scaler.center_[10]:.2f}mg")

    # PCA : garde 99.9% de la variance
    pca    = PCA(n_components=0.999, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X_scaled)
    n_comp = pca.n_components_
    var_total = pca.explained_variance_ratio_.sum() * 100

    ok(f"PCA : 25 features → {n_comp} composantes ({var_total:.2f}% variance)")
    for i, v in enumerate(pca.explained_variance_ratio_):
        info(f"  PC{i+1} : {v*100:.2f}%")

    return scaler, pca, X_pca


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — ENTRAÎNER LES 4 MODÈLES
# ══════════════════════════════════════════════════════════════════

def train_models(X_pca: np.ndarray, contamination: float) -> dict:
    head("ÉTAPE 4 — ENTRAÎNEMENT DES 4 MODÈLES")
    info(f"contamination={contamination} → {contamination*100:.1f}% supposés anormaux")
    info(f"Dataset : {X_pca.shape[0]} sessions × {X_pca.shape[1]} composantes PCA")

    trained = {}

    # ── 1. Isolation Forest ───────────────────────────────────────
    print(f"\n  {C}[1/4] Isolation Forest...{RS}")
    t0 = time.time()
    model_if = IsolationForest(
        n_estimators  = 300,          # 300 arbres → résultats stables
        contamination = contamination,
        max_samples   = "auto",
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    model_if.fit(X_pca)
    preds = model_if.predict(X_pca)
    n = int((preds == -1).sum())
    ok(f"IF — {time.time()-t0:.1f}s | {n} anomalies ({100*n/len(preds):.1f}%)")
    trained["if"] = model_if

    # ── 2. Local Outlier Factor ───────────────────────────────────
    print(f"\n  {C}[2/4] Local Outlier Factor...{RS}")
    t0 = time.time()
    # n_neighbors adapté à la taille du dataset
    n_neighbors = min(20, max(5, X_pca.shape[0] // 50))
    model_lof = LocalOutlierFactor(
        n_neighbors   = n_neighbors,
        contamination = contamination,
        novelty       = True,   # OBLIGATOIRE pour prédire sur nouvelles données
        n_jobs        = -1,
    )
    model_lof.fit(X_pca)
    preds = model_lof.predict(X_pca)
    n = int((preds == -1).sum())
    ok(f"LOF — {time.time()-t0:.1f}s | n_neighbors={n_neighbors} | {n} anomalies ({100*n/len(preds):.1f}%)")
    trained["lof"] = model_lof

    # ── 3. One-Class SVM ─────────────────────────────────────────
    print(f"\n  {C}[3/4] One-Class SVM...{RS}")
    t0 = time.time()
    # nu = contamination (proportion maximale d'outliers)
    # gamma="scale" → 1/(n_features * X.var()) — bon pour données normalisées
    model_ocsvm = OneClassSVM(
        nu    = min(0.5, contamination * 1.5),  # légèrement au-dessus pour tolérance
        kernel= "rbf",
        gamma = "scale",
    )
    # OCSVM lent sur grands datasets → sous-échantillonner si nécessaire
    if X_pca.shape[0] > 5000:
        idx = np.random.choice(X_pca.shape[0], 5000, replace=False)
        X_train_ocsvm = X_pca[idx]
        warn(f"OCSVM : dataset large → sous-échantillonnage à 5000 sessions")
    else:
        X_train_ocsvm = X_pca
    model_ocsvm.fit(X_train_ocsvm)
    preds = model_ocsvm.predict(X_pca)
    n = int((preds == -1).sum())
    ok(f"OCSVM — {time.time()-t0:.1f}s | {n} anomalies ({100*n/len(preds):.1f}%)")
    trained["ocsvm"] = model_ocsvm

    # ── 4. ECOD ─────────────────────────────────────────────────
    print(f"\n  {C}[4/4] ECOD...{RS}")
    t0 = time.time()
    try:
        from pyod.models.ecod import ECOD
        model_ecod = ECOD(contamination=contamination)
        model_ecod.fit(X_pca)
        preds = model_ecod.predict(X_pca)
        n = int((preds == 1).sum())
        ok(f"ECOD — {time.time()-t0:.1f}s | {n} anomalies ({100*n/len(preds):.1f}%)")
        trained["ecod"] = model_ecod
    except ImportError:
        warn("pyod non installé → ECOD remplacé par clone IF")
        warn("Pour installer : pip install pyod")
        model_ecod2 = IsolationForest(
            n_estimators  = 300,
            contamination = contamination,
            random_state  = RANDOM_STATE + 1,
            n_jobs        = -1,
        )
        model_ecod2.fit(X_pca)
        trained["ecod"] = model_ecod2
        ok(f"IF clone (fallback ECOD) — {time.time()-t0:.1f}s")

    return trained


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 5 — SAUVEGARDER
# ══════════════════════════════════════════════════════════════════

def save_models(scaler, pca, models: dict):
    head("ÉTAPE 5 — SAUVEGARDE")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "scaler_v3.pkl":     scaler,
        "pca_v3.pkl":        pca,
        "features_v3.pkl":   FEATURES_ORDER,
        "model_if_v3.pkl":   models["if"],
        "model_lof_v3.pkl":  models["lof"],
        "model_ocsvm_v3.pkl":models["ocsvm"],
        "model_ecod_v3.pkl": models["ecod"],
    }

    for filename, obj in files.items():
        path = MODEL_DIR / filename
        joblib.dump(obj, path)
        size = path.stat().st_size // 1024
        ok(f"Sauvegardé : {filename} ({size} KB)")


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 6 — VALIDATION RAPIDE
# ══════════════════════════════════════════════════════════════════

def validate(X_pca, models, scaler, pca, y_heuristic):
    head("ÉTAPE 6 — VALIDATION")

    results = {}
    for name, model in models.items():
        try:
            preds = model.predict(X_pca)
            # IF/LOF/OCSVM : -1=anomalie, +1=normal
            # ECOD          :  1=anomalie,  0=normal
            if name == "ecod":
                n_anom = int((preds == 1).sum())
            else:
                n_anom = int((preds == -1).sum())
            results[name] = n_anom
        except Exception as e:
            warn(f"{name} predict échoué : {e}")
            results[name] = -1

    info("Anomalies détectées par modèle :")
    for name, n in results.items():
        pct = 100 * n / len(X_pca) if n >= 0 else 0
        info(f"  {name.upper():6s} : {n:4d} / {len(X_pca)} ({pct:.1f}%)")

    # Test avec un point NORMAL connu (temp=30°C, vib=100mg)
    info("Test point NORMAL (temp=30°C, vib_z=100mg, vib_x=100mg) :")
    feat_test_normal = {f: 0.0 for f in FEATURES_ORDER}
    feat_test_normal.update({
        "temp_mean": 30.0, "temp_cur": 30.0, "temp_std": 0.5, "temp_trend": 0.0,
        "vib_z_mean": 100.0, "vib_z_cur": 100.0, "vib_z_std": 10.0,
        "vib_z_rms_w": 100.0, "vib_z_kurt": 3.0, "vib_z_crest": 1.4,
        "vib_x_mean": 100.0, "vib_x_std": 10.0, "vib_x_rms_w": 100.0, "vib_x_kurt": 3.0,
        "vib_y_mean": 100.0, "vib_y_std": 10.0, "vib_y_rms_w": 100.0, "vib_y_kurt": 3.0,
        "vib_total": 173.0, "health_score": 85.0,
    })
    X_n = np.array([[feat_test_normal[f] for f in FEATURES_ORDER]], dtype=np.float32)
    X_n = scaler.transform(X_n)
    X_n = pca.transform(X_n)

    votes_normal = 0
    for name, model in models.items():
        p = model.predict(X_n)[0]
        is_anom = (p == 1) if name == "ecod" else (p == -1)
        label   = "ANOMALY" if is_anom else "NORMAL"
        if label == "NORMAL":
            votes_normal += 1
        info(f"  {name.upper():6s} → {label}")
    info(f"  Vote : {votes_normal}/4 disent NORMAL")

    # Test avec un point ANORMAL (temp=58°C, vib=1200mg)
    info("Test point ANORMAL (temp=58°C, vib_z=1200mg) :")
    feat_test_anom = {f: 0.0 for f in FEATURES_ORDER}
    feat_test_anom.update({
        "temp_mean": 58.0, "temp_cur": 58.0, "temp_std": 2.0, "temp_trend": 0.5,
        "vib_z_mean": 1200.0, "vib_z_cur": 1200.0, "vib_z_std": 150.0,
        "vib_z_rms_w": 1210.0, "vib_z_kurt": 5.5, "vib_z_crest": 3.8,
        "vib_x_mean": 900.0, "vib_x_std": 120.0, "vib_x_rms_w": 910.0, "vib_x_kurt": 4.2,
        "vib_y_mean": 950.0, "vib_y_std": 110.0, "vib_y_rms_w": 955.0, "vib_y_kurt": 4.5,
        "vib_total": 1780.0, "health_score": 15.0,
    })
    X_a = np.array([[feat_test_anom[f] for f in FEATURES_ORDER]], dtype=np.float32)
    X_a = scaler.transform(X_a)
    X_a = pca.transform(X_a)

    votes_anom = 0
    for name, model in models.items():
        p = model.predict(X_a)[0]
        is_anom = (p == 1) if name == "ecod" else (p == -1)
        label   = "ANOMALY" if is_anom else "NORMAL"
        if label == "ANOMALY":
            votes_anom += 1
        info(f"  {name.upper():6s} → {label}")
    info(f"  Vote : {votes_anom}/4 disent ANOMALY")

    if votes_normal >= 3:
        ok("Point normal correctement classifié ✓")
    else:
        warn(f"Point normal : seulement {votes_normal}/4 NORMAL — contamination peut-être trop élevée")

    if votes_anom >= 2:
        ok("Point anormal correctement détecté ✓")
    else:
        warn(f"Point anormal : seulement {votes_anom}/4 ANOMALY — contamination peut-être trop basse")

    return votes_normal, votes_anom


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Réentraîne les 4 modèles ML sur les vraies données IFM"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help=f"Dataset CSV (défaut: {DEFAULT_CSV})")
    parser.add_argument("--contamination", type=float, default=CONTAMINATION,
                        help=f"Taux d'anomalies supposé (défaut: {CONTAMINATION})")
    args = parser.parse_args()

    print(f"\n{B}{C}{'═'*60}")
    print(f"  RÉENTRAÎNEMENT — Vraies données IFM")
    print(f"  Dataset       : {args.csv}")
    print(f"  Contamination : {args.contamination}")
    print(f"  Modèles → models/")
    print(f"{'═'*60}{RS}\n")

    t_total = time.time()

    # Pipeline complet
    df              = load_dataset(args.csv)
    X, y            = build_feature_matrix(df)
    scaler, pca, Xp = preprocess(X)
    models          = train_models(Xp, args.contamination)
    save_models(scaler, pca, models)
    validate(Xp, models, scaler, pca, y)

    elapsed = time.time() - t_total

    print(f"\n{B}{G}{'═'*60}")
    print(f"  ✅ RÉENTRAÎNEMENT TERMINÉ EN {elapsed:.0f}s")
    print(f"{'═'*60}{RS}")
    print(f"\n  {C}Relance l'API pour charger les nouveaux modèles :{RS}")
    print(f"  python api_unified_pythagore.py\n")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
