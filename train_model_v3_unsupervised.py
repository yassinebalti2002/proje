"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  train_model_v3_unsupervised.py                                             ║
║  Entraînement des 4 modèles non supervisés depuis MySQL ai_cp               ║
║                                                                              ║
║  Source de données : MySQL full_data (25 000 sessions réelles IFM)          ║
║  Modèles entraînés :                                                         ║
║    1. Isolation Forest  (IF)                                                 ║
║    2. Local Outlier Factor (LOF)                                             ║
║    3. One-Class SVM     (OCSVM)                                              ║
║    4. ECOD              (si pyod installé, sinon remplacé par IF clone)      ║
║                                                                              ║
║  Pipeline V6 (amélioré) :                                                    ║
║    SQL → parse JSON → sessions → features (31) → augmentation → RobustScaler║
║    → PCA(0.95) → cross-val 5-fold → entraînement 4 modèles                  ║
║    → vote majoritaire 2/4 → sauvegarde models/*.pkl → tests complets         ║
║                                                                              ║
║  Améliorations V6 :                                                          ║
║    - Contamination abaissée à 10% (moins de faux positifs)                   ║
║    - PCA 95% variance (meilleur débruitage)                                  ║
║    - Vote majoritaire 2/4 (meilleur rappel anomalies)                        ║
║    - 6 nouvelles features : FFT, delta-temporels, entropie, asymétrie        ║
║    - Data augmentation (×3 dataset)                                           ║
║    - Cross-validation 5-fold pour évaluation réelle                          ║
║    - OCSVM nu aligné sur contamination                                        ║
║                                                                              ║
║  Usage :                                                                     ║
║    python train_model_v3_unsupervised.py                                     ║
║    python train_model_v3_unsupervised.py --sql chemin/vers/fichier.sql       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import re
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import joblib
from scipy.stats import kurtosis, entropy as sp_entropy
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
)

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Chemin vers le fichier SQL (modifiable via --sql)
# Recherche automatique dans plusieurs emplacements
def _find_sql():
    candidates = [
        Path(__file__).parent / "ai_cp (5).sql",
        Path(__file__).parent.parent / "ai_cp (5).sql",
        Path.home() / "Desktop" / "ai_cp (5).sql",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "ai_cp (5).sql"  # fallback — affichera l'erreur si introuvable

DEFAULT_SQL = _find_sql()

# Dossier de sortie pour les modèles
MODEL_DIR = Path("models")  # relatif au dossier du script
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Connexion MySQL (pour lire full_data directement — meilleure source d'entraînement)
MYSQL_HOST     = "localhost"
MYSQL_PORT     = 3306
MYSQL_USER     = "root"
MYSQL_PASSWORD = "yassine2019"
MYSQL_DATABASE = "ai_cp"
MYSQL_TABLE    = "full_data"
MYSQL_SAMPLE_N = 25000   # sessions à extraire (1 session = 3 lignes full_data)

# Paramètres des modèles
CONTAMINATION   = 0.10   # V6 : abaissé à 10% (industriellement ~10% de pannes) — recalibré dynamiquement
WINDOW_SIZE     = 20     # V6 : fenêtre doublée (40s de données) pour capturer les tendances lentes
RANDOM_STATE    = 42
AUGMENT_FACTOR  = 3      # V6 : tripler le dataset par augmentation gaussienne
VOTE_THRESHOLD  = 2      # Vote majoritaire 2/4 — cohérent avec l'API (meilleur rappel)

# Seuils industriels IFM pour étiqueter les anomalies
# V6 : recalibrés sur données production IFM réelles (realtime_results.json)
# Echelle production : VibZ P50=322 P95=783 P99=1039 mg (vs SQL training: max=149 mg)
SEUIL_TEMP_MAX   = 50.4   # °C   → P95 production = 50.43°C (coherent)
SEUIL_VIB_MAX    = 783.0  # mg   → P95 production vib_z (au lieu de 7.3 qui etait le P95 SQL)
SEUIL_COURANT    = 97.0   # A    → P99 réel de la base (pas de données prod courante)
SEUIL_KURT_VIB   = 5.0    # kurtosis > 5 = impulsion anormale (bearing damage)
SEUIL_CREST_VIB  = 3.5    # crest factor > 3.5 = choc (normal < 2.5)
SEUIL_HEALTH_LOW = 45.0   # health_score < 45 = dégradation composite

# Chemin des données production (realtime_results.json)
REALTIME_RESULTS = "realtime_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# COULEURS TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════
G  = "\033[92m"   # vert
R  = "\033[91m"   # rouge
Y  = "\033[93m"   # jaune
C  = "\033[96m"   # cyan
B  = "\033[1m"    # gras
RS = "\033[0m"    # reset

def ok(msg):  print(f"  {G}✅ {msg}{RS}")
def err(msg): print(f"  {R}❌ {msg}{RS}")
def info(msg):print(f"  {C}ℹ  {msg}{RS}")
def warn(msg):print(f"  {Y}⚠  {msg}{RS}")
def head(msg):print(f"\n{B}{C}{'═'*60}\n  {msg}\n{'═'*60}{RS}")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — LECTURE ET PARSING DU FICHIER SQL
# ═══════════════════════════════════════════════════════════════════════════════

def parse_sql_to_dataframe(sql_path: str) -> pd.DataFrame:
    """
    Lit le fichier SQL et extrait les données de la table motor_mesure.

    La table motor_mesure contient des mesures capteurs IFM déjà consolidées :
      id | motor_id | id_cp | date | temperature | x | y | z | vibration_totale | courant

    Chaque ligne = une mesure complète d'un capteur à un instant donné.
    C'est la table la plus propre pour l'entraînement (353 lignes réelles).

    On utilise aussi motor_measurements si motor_mesure est insuffisante.
    """
    head("ÉTAPE 1 — LECTURE DU FICHIER SQL")
    info(f"Fichier : {sql_path}")
    info(f"Taille  : {Path(sql_path).stat().st_size / 1e6:.1f} MB")

    print("  Lecture en cours...", end="", flush=True)
    content = open(sql_path, encoding='utf-8', errors='ignore').read()
    print(f" {G}OK{RS}")

    rows = []

    # ── Parser motor_mesure ────────────────────────────────────────────────────
    # Format : (id, motor_id, 'id_cp', 'date', temperature, x, y, z, vib_tot, courant, 'alert')
    pattern_mesure = (
        r"\((\d+),\s*(\d+),\s*'([^']+)',\s*'([^']+)',\s*"
        r"([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*"
        r"([\d.]+),\s*([\d.]+),\s*'?([^',)]*)'?\)"
    )

    # On cherche uniquement dans le bloc INSERT de motor_mesure
    # pour éviter de confondre avec d'autres tables
    bloc_mesure = re.search(
        r"INSERT INTO `motor_mesure`.*?(?=INSERT INTO `\w|$)",
        content, re.DOTALL
    )

    if bloc_mesure:
        for m in re.finditer(pattern_mesure, bloc_mesure.group()):
            try:
                rows.append({
                    'source':          'motor_mesure',
                    'sensor_id':       m.group(3).upper(),
                    'timestamp':       m.group(4),
                    'temperature':     float(m.group(5)),
                    'vibration_x':     float(m.group(6)),
                    'vibration_y':     float(m.group(7)),
                    'vibration_z':     float(m.group(8)),
                    'vibration_total': float(m.group(9)),
                    'current':         float(m.group(10)),
                })
            except (ValueError, IndexError):
                continue
        ok(f"motor_mesure : {len(rows)} mesures extraites")
    else:
        warn("Bloc motor_mesure introuvable dans le SQL")

    # ── Parser motor_measurements (source complémentaire) ─────────────────────
    # Format : (measurement_id, motor_id, 'timestamp', temperature, courant,
    #           vibration, acceleration, thdi, thdu, vitesse, cosphi, 'Alert_Status', ...)
    rows_mm = []
    bloc_mm = re.search(
        r"INSERT INTO `motor_measurements`.*?(?=INSERT INTO `\w|$)",
        content, re.DOTALL
    )
    if bloc_mm:
        pattern_mm = (
            r"\((\d+),\s*(\d+),\s*'([^']+)',\s*([\d.]+),\s*([\d.]+),\s*"
            r"([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*"
            r"([\d.]+),\s*'([^']+)'"
        )
        for m in re.finditer(pattern_mm, bloc_mm.group()):
            try:
                rows_mm.append({
                    'source':          'motor_measurements',
                    'sensor_id':       f"motor_{m.group(2)}",
                    'timestamp':       m.group(3),
                    'temperature':     float(m.group(4)),
                    'vibration_x':     float(m.group(6)) * 100,  # convertir en mg
                    'vibration_y':     float(m.group(6)) * 80,
                    'vibration_z':     float(m.group(6)) * 120,
                    'vibration_total': float(m.group(6)) * 200,
                    'current':         float(m.group(5)),
                    'alert_status':    m.group(12),
                })
            except (ValueError, IndexError):
                continue
        ok(f"motor_measurements : {len(rows_mm)} mesures extraites")
        rows.extend(rows_mm)

    if not rows:
        err("Aucune donnée extraite du SQL !")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['temperature', 'vibration_z'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    ok(f"DataFrame total : {len(df)} lignes | {df['sensor_id'].nunique()} capteurs uniques")
    info(f"Capteurs : {sorted(df['sensor_id'].unique())[:10]}...")
    info(f"Période  : {df['timestamp'].min()} → {df['timestamp'].max()}")
    info(f"Température : {df['temperature'].min():.1f}°C – {df['temperature'].max():.1f}°C")
    info(f"Vib Z       : {df['vibration_z'].min():.2f} – {df['vibration_z'].max():.2f} mg")
    info(f"Courant     : {df['current'].min():.1f} – {df['current'].max():.1f} A")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1b — LECTURE DES DONNÉES PRODUCTION (realtime_results.json)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_features_from_realtime(path: str) -> tuple:
    """
    Lit realtime_results.json et extrait DIRECTEMENT les features déjà calculées.

    Pourquoi cette approche est correcte :
    - realtime_results.json stocke les features calculées par l'API sur de VRAIES
      fenêtres de 20 mesures (résolution temporelle correcte : 2s entre mesures)
    - Reconstruire des fenêtres depuis les mesures brutes donne une résolution
      différente (~38s entre entrées) → features temp_trend, delta_vib faussées
    - En utilisant les features stockées, l'entraînement et l'inférence utilisent
      exactement la même distribution de features

    Labelling heuristique physique (indépendant de l'ancien modèle) :
    - vib_total > 1039 (P99 production) → ANOMALIE certaine
    - vib_z_rms_w > 650   → ANOMALIE (P90)
    - health_score < 60   → ANOMALIE composite
    - temp_mean > 50      → surchauffe
    - vib_z_kurt > 5      → choc impulsif
    """
    p = Path(path)
    if not p.exists():
        warn(f"{path} introuvable")
        return np.array([]), [], np.array([])

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        warn(f"Erreur lecture {path} : {e}")
        return np.array([]), [], np.array([])

    FEAT_KEYS = [
        'temp_mean', 'temp_std', 'temp_trend', 'temp_cur',
        'vib_z_mean', 'vib_z_std', 'vib_z_rms_w', 'vib_z_kurt', 'vib_z_crest', 'vib_z_cur',
        'vib_x_mean', 'vib_x_std', 'vib_x_rms_w', 'vib_x_kurt',
        'vib_y_mean', 'vib_y_std', 'vib_y_rms_w', 'vib_y_kurt',
        'vib_total', 'health_score',
        'acc_p2p', 'acc_z2p', 'acc_crest', 'acc_rms',
        'current_mean',
        'delta_vib', 'delta_temp', 'vib_entropy', 'fft_ratio',
        'vib_asym_xy', 'vib_asym_xz',
    ]

    X_rows, y_rows = [], []
    for entry in data:
        feat = (entry.get("predict") or {}).get("features") or {}
        if not feat:
            continue
        row = [float(feat.get(k, 0.0) or 0.0) for k in FEAT_KEYS]
        if any(np.isnan(v) or np.isinf(v) for v in row):
            row = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in row]

        # Labelling heuristique physique (seuils production calibrés)
        vib_total   = feat.get("vib_total", 0) or 0
        vib_z_rms   = feat.get("vib_z_rms_w", 0) or 0
        health      = feat.get("health_score", 100) or 100
        temp        = feat.get("temp_mean", 0) or 0
        kurt        = feat.get("vib_z_kurt", 0) or 0
        crest       = feat.get("vib_z_crest", 0) or 0

        is_anom = (
            vib_total > 1039        # P99 production
            or vib_z_rms > 650      # P90 production
            or health < 60          # degradation composite severe
            or temp > 50.4          # surchauffe P95
            or kurt > 5.0           # choc impulsif roulement
            or crest > 3.5          # facteur de crête
        )
        X_rows.append(row)
        y_rows.append(1 if is_anom else 0)

    if not X_rows:
        warn("Aucune feature extraite de realtime_results.json")
        return np.array([]), [], np.array([])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=int)
    X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)

    n_anom = int(y.sum())
    ok(f"Features production : {len(X)} vecteurs × {len(FEAT_KEYS)} features | {n_anom} anomalies ({n_anom/len(y)*100:.1f}%)")
    return X, FEAT_KEYS, y


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — CONSTRUCTION DES SESSIONS ET EXTRACTION DES 25 FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def rms(arr):
    """Root Mean Square — mesure l'énergie du signal."""
    return float(np.sqrt(np.mean(np.array(arr, dtype=float)**2)))

def crest_factor(arr):
    """Facteur de crête = max / RMS — détecte les chocs impulsionnels."""
    r = rms(arr)
    return float(np.max(np.abs(arr)) / r) if r > 0 else 1.0

def trend(arr):
    """
    Pente de la régression linéaire sur la fenêtre.
    Une pente positive = signal en hausse = dégradation potentielle.
    """
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    p = np.polyfit(x, np.array(arr, dtype=float), 1)
    return float(p[0])

def signal_entropy(arr: np.ndarray) -> float:
    """Entropie de Shannon du signal — mesure l'irregularite/complexite."""
    counts, _ = np.histogram(arr, bins=10)
    return float(sp_entropy(counts + 1e-9))


def fft_dominant_freq_ratio(arr: np.ndarray) -> float:
    """Ratio energie frequence dominante / energie totale FFT — detecte periodicite anormale."""
    if len(arr) < 4:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(arr))
    total_energy = np.sum(fft_vals**2) + 1e-9
    dominant_energy = np.max(fft_vals**2)
    return float(dominant_energy / total_energy)


def extract_features_from_window(window: pd.DataFrame) -> dict:
    """
    Calcule les 31 features depuis une fenêtre de WINDOW_SIZE mesures.

    V6 : +6 nouvelles features (FFT, delta-temporels, entropie, asymetrie axes)
    Ces features sont exactement les mêmes que celles utilisées par
    l'API (features_v3.pkl) — indispensable pour la compatibilité.

    Groupes de features :
      - Température (4) : mean, std, trend, valeur courante
      - Vibration Z  (6) : mean, std, rms, kurtosis, crest, valeur courante
      - Vibration X  (4) : mean, std, rms, kurtosis
      - Vibration Y  (4) : mean, std, rms, kurtosis
      - Combinées    (2) : vib_total (norme vectorielle), health_score
      - Accélération (4) : p2p, z2p, crest, rms  (calculés depuis vib_z)
      - Courant      (1) : mean
      - V6 nouvelles (6) : entropie, FFT ratio, delta_vib, delta_temp,
                           asymetrie XY, asymetrie XZ
    """
    T  = window['temperature'].values.astype(float)
    VX = window['vibration_x'].values.astype(float)
    VY = window['vibration_y'].values.astype(float)
    VZ = window['vibration_z'].values.astype(float)
    I  = window['current'].values.astype(float)

    # Norme vectorielle 3D à chaque instant : √(X²+Y²+Z²)
    VT = np.sqrt(VX**2 + VY**2 + VZ**2)

    # Health score V6 : formule alignée avec l'API
    VIB_TOTAL_MAX = float(np.sqrt(3) * 1500)
    temp_n  = max(0.0, min(1.0, (np.mean(T)  - 25) / (65 - 25 + 1e-9)))
    vib_n   = max(0.0, min(1.0, np.mean(VT)         / (VIB_TOTAL_MAX + 1e-9)))
    kurt_n  = max(0.0, min(1.0, (kurtosis(VZ, fisher=False) if len(VZ)>3 else 3.0) / 10.0))
    cur_n   = max(0.0, min(1.0, np.mean(I)           / (200 + 1e-9)))
    health  = round(100 * (1 - 0.35*temp_n - 0.35*vib_n - 0.30*kurt_n), 1)

    # Accélération estimée depuis vib_z (dérivée approx)
    acc = np.diff(VZ, prepend=VZ[0])
    acc_p2p = float(np.max(acc) - np.min(acc))
    acc_z2p = float(np.max(np.abs(acc)))
    acc_c   = crest_factor(acc + 1e-9)
    acc_r   = rms(acc)

    # ── V6 : nouvelles features ──────────────────────────────────────────
    # Delta inter-fenetres : variation entre premiere et deuxieme moitie
    mid = len(VZ) // 2
    delta_vib  = float(np.mean(VZ[mid:]) - np.mean(VZ[:mid]))   # tendance intra-fenetre
    delta_temp = float(np.mean(T[mid:])  - np.mean(T[:mid]))

    # Entropie de Shannon sur vib_z (irregularite du signal)
    vib_entropy = signal_entropy(VZ)

    # Ratio FFT : periodicite anormale (choc repetitif = defaut roulement)
    fft_ratio = fft_dominant_freq_ratio(VZ)

    # Asymetrie inter-axes : desequilibre mecanique
    vib_asym_xy = float(abs(np.mean(VX) - np.mean(VY)) / (np.mean(VX) + np.mean(VY) + 1e-9))
    vib_asym_xz = float(abs(np.mean(VX) - np.mean(VZ)) / (np.mean(VX) + np.mean(VZ) + 1e-9))

    feats = {
        # ── Température (4 features) ──────────────────────────────────────
        'temp_mean':    float(np.mean(T)),
        'temp_std':     float(np.std(T)),
        'temp_trend':   trend(T),
        'temp_cur':     float(T[-1]),

        # ── Vibration Z — axe principal (6 features) ──────────────────────
        'vib_z_mean':   float(np.mean(VZ)),
        'vib_z_std':    float(np.std(VZ)),
        'vib_z_rms_w':  rms(VZ),
        'vib_z_kurt':   float(kurtosis(VZ, fisher=False) if len(VZ)>3 else 3.0),
        'vib_z_crest':  crest_factor(VZ + 1e-9),
        'vib_z_cur':    float(VZ[-1]),

        # ── Vibration X (4 features) ──────────────────────────────────────
        'vib_x_mean':   float(np.mean(VX)),
        'vib_x_std':    float(np.std(VX)),
        'vib_x_rms_w':  rms(VX),
        'vib_x_kurt':   float(kurtosis(VX, fisher=False) if len(VX)>3 else 3.0),

        # ── Vibration Y (4 features) ──────────────────────────────────────
        'vib_y_mean':   float(np.mean(VY)),
        'vib_y_std':    float(np.std(VY)),
        'vib_y_rms_w':  rms(VY),
        'vib_y_kurt':   float(kurtosis(VY, fisher=False) if len(VY)>3 else 3.0),

        # ── Combinées (2 features) ────────────────────────────────────────
        'vib_total':    float(np.mean(VT)),
        'health_score': health,

        # ── Accélération (4 features) ─────────────────────────────────────
        'acc_p2p':      acc_p2p,
        'acc_z2p':      acc_z2p,
        'acc_crest':    acc_c,
        'acc_rms':      acc_r,

        # ── Courant électrique (1 feature) ────────────────────────────────
        'current_mean': float(np.mean(I)),

        # ── V6 : 6 nouvelles features ─────────────────────────────────────
        'delta_vib':    delta_vib,    # variation intra-fenetre vib_z
        'delta_temp':   delta_temp,   # variation intra-fenetre temperature
        'vib_entropy':  vib_entropy,  # irregularite du signal
        'fft_ratio':    fft_ratio,    # periodicite anormale
        'vib_asym_xy':  vib_asym_xy,  # desequilibre axial XY
        'vib_asym_xz':  vib_asym_xz,  # desequilibre axial XZ
    }
    return feats


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Construit la matrice de features X depuis le DataFrame.

    Stratégie : fenêtre glissante par capteur.
    Pour chaque capteur, on prend des fenêtres de WINDOW_SIZE mesures
    consécutives et on calcule les 31 features de chaque fenêtre.

    Retourne : (X, feature_names, labels_heuristiques)
    """
    head("ÉTAPE 2 — CONSTRUCTION DES FEATURES")

    FEATURES_ORDER = [
        'temp_mean', 'temp_std', 'temp_trend', 'temp_cur',
        'vib_z_mean', 'vib_z_std', 'vib_z_rms_w', 'vib_z_kurt', 'vib_z_crest', 'vib_z_cur',
        'vib_x_mean', 'vib_x_std', 'vib_x_rms_w', 'vib_x_kurt',
        'vib_y_mean', 'vib_y_std', 'vib_y_rms_w', 'vib_y_kurt',
        'vib_total', 'health_score',
        'acc_p2p', 'acc_z2p', 'acc_crest', 'acc_rms',
        'current_mean',
        # V6 : nouvelles features
        'delta_vib', 'delta_temp', 'vib_entropy', 'fft_ratio',
        'vib_asym_xy', 'vib_asym_xz',
    ]

    all_features = []
    heuristic_labels = []   # 1 = anomalie heuristique, 0 = normal

    sensors = df['sensor_id'].unique()
    info(f"Traitement de {len(sensors)} capteurs avec fenêtre de {WINDOW_SIZE} mesures")

    for sid in sensors:
        sensor_df = df[df['sensor_id'] == sid].reset_index(drop=True)

        # Besoin d'au moins WINDOW_SIZE mesures pour une fenêtre
        if len(sensor_df) < WINDOW_SIZE:
            warn(f"Capteur {sid} : seulement {len(sensor_df)} mesures → ignoré")
            continue

        # Fenêtre glissante : fenêtre [i : i+WINDOW_SIZE]
        for i in range(len(sensor_df) - WINDOW_SIZE + 1):
            window = sensor_df.iloc[i : i + WINDOW_SIZE]
            try:
                feats = extract_features_from_window(window)
                row = [feats[f] for f in FEATURES_ORDER]
                all_features.append(row)

                # Étiquette heuristique — 6 conditions (seuils industriels IFM)
                is_anomaly = (
                    feats['temp_mean']    > SEUIL_TEMP_MAX    # surchauffe
                    or feats['vib_z_rms_w']  > SEUIL_VIB_MAX  # vibration RMS
                    or feats['current_mean'] > SEUIL_COURANT   # surcourant
                    or feats['health_score'] < SEUIL_HEALTH_LOW  # dégradation composite
                    or feats['vib_z_kurt']   > SEUIL_KURT_VIB   # choc impulsif bearing
                    or feats['vib_z_crest']  > SEUIL_CREST_VIB  # facteur de crête
                )
                heuristic_labels.append(1 if is_anomaly else 0)
            except Exception as e:
                continue

    if not all_features:
        warn("Pas assez de fenêtres glissantes — utilisation des mesures individuelles")
        for _, row in df.iterrows():
            feats = {
                'temp_mean': row['temperature'], 'temp_std': 0.1,
                'temp_trend': 0.0, 'temp_cur': row['temperature'],
                'vib_z_mean': row['vibration_z'], 'vib_z_std': 0.1,
                'vib_z_rms_w': row['vibration_z'], 'vib_z_kurt': 3.0,
                'vib_z_crest': 1.4, 'vib_z_cur': row['vibration_z'],
                'vib_x_mean': row['vibration_x'], 'vib_x_std': 0.1,
                'vib_x_rms_w': row['vibration_x'], 'vib_x_kurt': 3.0,
                'vib_y_mean': row['vibration_y'], 'vib_y_std': 0.1,
                'vib_y_rms_w': row['vibration_y'], 'vib_y_kurt': 3.0,
                'vib_total': np.sqrt(row['vibration_x']**2 + row['vibration_y']**2 + row['vibration_z']**2),
                'health_score': max(0, 100 - max(0, row['temperature'] - 35)*3),
                'acc_p2p': 0.0, 'acc_z2p': 0.0, 'acc_crest': 1.0, 'acc_rms': 0.0,
                'current_mean': row['current'],
                'delta_vib': 0.0, 'delta_temp': 0.0,
                'vib_entropy': 1.0, 'fft_ratio': 0.1,
                'vib_asym_xy': 0.0, 'vib_asym_xz': 0.0,
            }
            all_features.append([feats[f] for f in FEATURES_ORDER])
            is_anom = (row['temperature'] > SEUIL_TEMP_MAX or
                       row['vibration_z'] > SEUIL_VIB_MAX or
                       row['current'] > SEUIL_COURANT)
            heuristic_labels.append(1 if is_anom else 0)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(heuristic_labels, dtype=int)
    X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)

    # ── V6 : Data augmentation (×AUGMENT_FACTOR) ─────────────────────────
    # Ajoute du bruit gaussien faible (2%) sur chaque échantillon normal
    # pour enrichir le dataset et améliorer la généralisation
    rng = np.random.default_rng(RANDOM_STATE)
    X_aug_list = [X]
    y_aug_list = [y]
    for _ in range(AUGMENT_FACTOR - 1):
        noise = rng.normal(0, 0.02, size=X.shape).astype(np.float32)
        X_noisy = np.clip(X + X * noise, 0, None)
        X_aug_list.append(X_noisy)
        y_aug_list.append(y)
    X = np.vstack(X_aug_list)
    y = np.concatenate(y_aug_list)
    X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)

    n_anom = int(y.sum())
    ok(f"Matrice originale  : {len(all_features)} sessions × {len(FEATURES_ORDER)} features")
    ok(f"Après augmentation : {X.shape[0]} sessions × {X.shape[1]} features (×{AUGMENT_FACTOR})")
    ok(f"Anomalies heuristiques : {n_anom} ({n_anom/len(y)*100:.1f}%)")
    info(f"Features ({len(FEATURES_ORDER)}) : {FEATURES_ORDER}")

    return X, FEATURES_ORDER, y


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — PRÉTRAITEMENT : SCALER + PCA
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(X: np.ndarray) -> tuple:
    """
    Applique RobustScaler puis PCA sur la matrice de features.

    RobustScaler :
      - Centre par la médiane (pas la moyenne) → résistant aux outliers
      - Met à l'échelle par l'IQR (interquartile range)
      - Idéal pour les données industrielles avec pics occasionnels

    PCA :
      - Réduit la dimensionnalité en gardant les directions de variance max
      - n_components=0.999 → garde 99.9% de la variance
      - En pratique : 25 features → 5 composantes principales
      - Réduit le bruit et accélère les modèles
    """
    head("ÉTAPE 3 — PRÉTRAITEMENT (Scaler + PCA)")

    # RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    ok(f"RobustScaler appliqué → shape {X_scaled.shape}")

    # PCA — V6 : 0.95 au lieu de 0.999 pour un meilleur débruitage
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    ok(f"PCA appliqué → {X_pca.shape[1]} composantes ({pca.explained_variance_ratio_.sum()*100:.2f}% variance retenue)")
    info("V6 : PCA(0.95) au lieu de 0.999 — filtre le bruit résiduel")

    for i, ev in enumerate(pca.explained_variance_ratio_):
        info(f"  PC{i+1} : {ev*100:.2f}%")

    return scaler, pca, X_pca


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — ENTRAÎNEMENT DES 4 MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

def train_models(X_pca: np.ndarray) -> dict:
    """
    Entraîne les 4 modèles de détection d'anomalies non supervisés.

    Tous les modèles sont NON SUPERVISÉS : ils apprennent ce qu'est
    le comportement NORMAL sans jamais voir d'étiquettes.

    ┌─────────────────┬──────────────────────────────────────────────┐
    │ Modèle          │ Principe                                     │
    ├─────────────────┼──────────────────────────────────────────────┤
    │ Isolation Forest│ Isole les points rares par des arbres random │
    │ LOF             │ Compare densité locale aux k voisins          │
    │ One-Class SVM   │ Frontière hypersphérique autour du normal     │
    │ ECOD            │ Queues de distribution empiriques (si pyod)   │
    └─────────────────┴──────────────────────────────────────────────┘

    contamination=0.10 signifie qu'on suppose que ~10% des données
    d'entraînement sont déjà anormales (recalibré dynamiquement dans main).
    """
    head("ÉTAPE 4 — ENTRAÎNEMENT DES 4 MODÈLES")

    trained = {}

    # ── Modèle 1 : Isolation Forest ───────────────────────────────────────────
    print(f"\n  {C}[1/4] Isolation Forest...{RS}")
    t0 = time.time()
    # n_estimators=200 : 200 arbres → résultats stables
    # max_samples='auto' : √n échantillons par arbre
    model_if = IsolationForest(
        n_estimators  = 200,
        contamination = CONTAMINATION,
        max_samples   = 'auto',
        random_state  = RANDOM_STATE,
        n_jobs        = -1,   # utilise tous les CPU
    )
    model_if.fit(X_pca)
    preds_if = model_if.predict(X_pca)
    # IF retourne -1 pour anomalie, +1 pour normal
    n_anom = int((preds_if == -1).sum())
    ok(f"IF entraîné en {time.time()-t0:.2f}s | {n_anom} anomalies ({n_anom/len(preds_if)*100:.1f}%)")
    trained['if'] = model_if

    # ── Modèle 2 : Local Outlier Factor ───────────────────────────────────────
    print(f"\n  {C}[2/4] Local Outlier Factor...{RS}")
    t0 = time.time()
    # novelty=True : permet d'utiliser .predict() sur de nouvelles données
    # n_neighbors=20 : compare chaque point à ses 20 voisins les plus proches
    model_lof = LocalOutlierFactor(
        n_neighbors   = 20,  # sqrt(~588) ≈ 24, recommandation standard
        contamination = CONTAMINATION,
        novelty       = True,   # IMPORTANT : sinon pas de .predict()
        n_jobs        = -1,
    )
    model_lof.fit(X_pca)
    preds_lof = model_lof.predict(X_pca)
    n_anom = int((preds_lof == -1).sum())
    ok(f"LOF entraîné en {time.time()-t0:.2f}s | {n_anom} anomalies ({n_anom/len(preds_lof)*100:.1f}%)")
    trained['lof'] = model_lof

    # ── Modèle 3 : One-Class SVM ──────────────────────────────────────────────
    print(f"\n  {C}[3/4] One-Class SVM...{RS}")
    t0 = time.time()
    # V6 : nu aligné sur CONTAMINATION (cohérent avec IF et LOF)
    model_ocsvm = OneClassSVM(
        kernel = 'rbf',
        nu     = CONTAMINATION,   # V6 : aligné sur contamination globale
        gamma  = 'scale',
    )
    model_ocsvm.fit(X_pca)
    preds_ocsvm = model_ocsvm.predict(X_pca)
    n_anom = int((preds_ocsvm == -1).sum())
    ok(f"OCSVM entraîné en {time.time()-t0:.2f}s | {n_anom} anomalies ({n_anom/len(preds_ocsvm)*100:.1f}%)")
    trained['ocsvm'] = model_ocsvm

    # ── Modèle 4 : ECOD ou IsolationForest clone ──────────────────────────────
    print(f"\n  {C}[4/4] ECOD...{RS}")
    t0 = time.time()
    try:
        from pyod.models.ecod import ECOD
        model_ecod = ECOD(contamination=CONTAMINATION)
        model_ecod.fit(X_pca)
        preds_ecod = model_ecod.predict(X_pca)
        n_anom = int((preds_ecod == 1).sum())   # ECOD retourne 1 pour anomalie
        ok(f"ECOD (pyod) entraîné en {time.time()-t0:.2f}s | {n_anom} anomalies")
        trained['ecod'] = model_ecod
        trained['ecod_type'] = 'pyod'
    except ImportError:
        warn("pyod non disponible → ECOD remplacé par IsolationForest (paramètres différents)")
        # Clone IF avec paramètres légèrement différents pour apporter de la diversité
        model_ecod = IsolationForest(
            n_estimators  = 150,
            contamination = CONTAMINATION,
            max_features  = 0.8,     # sous-espace aléatoire
            random_state  = RANDOM_STATE + 1,
            n_jobs        = -1,
        )
        model_ecod.fit(X_pca)
        preds_ecod = model_ecod.predict(X_pca)
        n_anom = int((preds_ecod == -1).sum())
        ok(f"ECOD-clone (IF) entraîné en {time.time()-t0:.2f}s | {n_anom} anomalies")
        trained['ecod'] = model_ecod
        trained['ecod_type'] = 'if_clone'

    return trained


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — TESTS ET ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_models(trained: dict, X_pca: np.ndarray, y_true: np.ndarray,
                    scaler, pca, feature_names: list) -> dict:
    """
    Évalue chaque modèle (seuil fixe + seuil optimal) et le soft-voting ensemble.
    Labels de référence = étiquettes heuristiques industrielles.
    """
    from sklearn.preprocessing import MinMaxScaler as MMS

    head("ÉTAPE 5 — ÉVALUATION DES MODÈLES (V6 : cross-val 5-fold + vote 2/4)")

    # ── Scores continus (anomalie = valeur haute) ──────────────────────────────
    scores_if    = -trained['if'].score_samples(X_pca)
    scores_lof   = -trained['lof'].score_samples(X_pca)
    scores_ocsvm = -trained['ocsvm'].decision_function(X_pca)
    ecod_type = trained.get('ecod_type', 'pyod')
    if ecod_type == 'pyod':
        scores_ecod = trained['ecod'].decision_scores_
    else:
        scores_ecod = -trained['ecod'].score_samples(X_pca)

    # ── Seuil optimal par modèle (sweep 1%-55% → maximise F1) ─────────────────
    def best_threshold_f1(scores, y_true):
        best_f1, best_preds = 0.0, np.zeros(len(y_true), dtype=int)
        for pct in range(1, 56):
            thr  = np.percentile(scores, 100 - pct)
            pred = (scores >= thr).astype(int)
            f1   = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_preds = f1, pred.copy()
        return best_preds, best_f1

    opt_if,    f1_opt_if    = best_threshold_f1(scores_if,    y_true)
    opt_lof,   f1_opt_lof   = best_threshold_f1(scores_lof,   y_true)
    opt_ocsvm, f1_opt_ocsvm = best_threshold_f1(scores_ocsvm, y_true)
    opt_ecod,  f1_opt_ecod  = best_threshold_f1(scores_ecod,  y_true)

    # ── Soft-voting ensemble : moyenne IF+OCSVM+ECOD (LOF exclu, trop faible) ──
    def norm(s): return MMS().fit_transform(s.reshape(-1,1)).flatten()
    avg_score = (norm(scores_if) + norm(scores_ocsvm) + norm(scores_ecod)) / 3.0
    opt_soft, f1_opt_soft = best_threshold_f1(avg_score, y_true)

    # ── Prédictions binaires à contamination fixe (pour comparaison) ──────────
    def to_bin_if(p):   return (p == -1).astype(int)
    def to_bin_ecod(p): return (p == 1).astype(int) if ecod_type=='pyod' else (p==-1).astype(int)

    preds_if    = to_bin_if(trained['if'].predict(X_pca))
    preds_lof   = to_bin_if(trained['lof'].predict(X_pca))
    preds_ocsvm = to_bin_if(trained['ocsvm'].predict(X_pca))
    preds_ecod  = to_bin_ecod(trained['ecod'].predict(X_pca))
    votes        = preds_if + preds_lof + preds_ocsvm + preds_ecod
    preds_vote2  = (votes >= 2).astype(int)
    preds_vote3  = (votes >= 3).astype(int)   # V6 : vote majoritaire strict recommandé

    # ── V6 : Cross-validation 5-fold ─────────────────────────────────────────
    print(f"\n  {C}[CV] Cross-validation 5-fold (IF — référence)...{RS}")
    from sklearn.base import BaseEstimator, ClassifierMixin
    class IFWrapper(BaseEstimator, ClassifierMixin):
        """Wrapper pour utiliser IF dans cross_val_score."""
        def __init__(self, contamination=0.10, n_estimators=200):
            self.contamination = contamination
            self.n_estimators = n_estimators
        def fit(self, X, y=None):
            self.model_ = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=RANDOM_STATE, n_jobs=-1)
            self.model_.fit(X)
            return self
        def predict(self, X):
            return (self.model_.predict(X) == -1).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    try:
        cv_scores = cross_val_score(IFWrapper(CONTAMINATION), X_pca, y_true,
                                    cv=cv, scoring='f1')
        # Filtrer les NaN (peuvent survenir quand un fold n'a pas les 2 classes)
        valid = cv_scores[~np.isnan(cv_scores)]
        if len(valid) > 0:
            ok(f"Cross-val F1 (IF) : {valid.mean():.4f} +/- {valid.std():.4f}  ({len(valid)}/{len(cv_scores)} folds valides)")
        else:
            warn("Cross-val F1 indisponible (labels heuristiques non stratifiables sur ce split)")
        info(f"Scores par fold : {[f'{s:.4f}' if not np.isnan(s) else 'N/A' for s in cv_scores]}")
    except Exception as e:
        warn(f"Cross-val echouee : {e}")

    # ── Tableau comparatif ────────────────────────────────────────────────────
    metrics = {}
    models_preds = {
        'IF':        preds_if,
        'LOF':       preds_lof,
        'OCSVM':     preds_ocsvm,
        'ECOD':      preds_ecod,
        'Vote2/4':   preds_vote2,
        'Vote3/4':   preds_vote3,
        'IF_opt':    opt_if,
        'LOF_opt':   opt_lof,
        'OCSVM_opt': opt_ocsvm,
        'ECOD_opt':  opt_ecod,
        'SoftVote':  opt_soft,
    }

    print(f"\n  {'Modèle':<12} {'F1':>6} {'Précision':>10} {'Rappel':>8} {'AUC-ROC':>8} {'Anomalies':>10}")
    print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*10}")

    for name, preds in models_preds.items():
        f1   = f1_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_true, preds)
        except Exception:
            auc = 0.5
        n_a  = int(preds.sum())
        acc  = accuracy_score(y_true, preds)
        color = G if f1 > 0.75 else (Y if f1 > 0.5 else R)
        print(f"  {name:<12} {color}{f1:>6.4f}{RS} {prec:>10.4f} {rec:>8.4f} {auc:>8.4f} {n_a:>10d}")
        metrics[name] = {'f1': f1, 'precision': prec, 'recall': rec, 'auc_roc': auc, 'n_anomalies': n_a, 'accuracy': acc}

    # Stocker scores et seuils optimaux pour la sauvegarde (préfixe _ = interne)
    def _get_thr(scores, n_anom):
        return float(np.percentile(scores, 100 - n_anom / len(scores) * 100))

    metrics['_opt_preds']   = opt_soft
    metrics['_avg_score']   = avg_score
    metrics['_f1_softvote'] = f1_opt_soft
    metrics['_opt_thresholds'] = {
        'if_threshold':    _get_thr(scores_if,    int(opt_if.sum())),
        'lof_threshold':   _get_thr(scores_lof,   int(opt_lof.sum())),
        'ocsvm_threshold': _get_thr(scores_ocsvm, int(opt_ocsvm.sum())),
        'ecod_threshold':  _get_thr(scores_ecod,  int(opt_ecod.sum())),
    }

    # ── Test sur cas extrêmes ──────────────────────────────────────────────────
    head("TEST CAS EXTRÊMES")

    test_cases = [
        {
            'name':  'NORMAL — moteur sain',
            'data':  {'temperature': 35.0, 'vibration_x': 2.5, 'vibration_y': 2.0,
                      'vibration_z': 3.0, 'vibration_total': 4.5, 'current': 15.0},
            'expect': 'NORMAL'
        },
        {
            'name':  'CRITIQUE — surchauffe + vibration',
            'data':  {'temperature': 62.0, 'vibration_x': 18.0, 'vibration_y': 16.0,
                      'vibration_z': 19.0, 'vibration_total': 30.0, 'current': 250.0},
            'expect': 'ANOMALY'
        },
        {
            'name':  'FRONTIÈRE — temp limite',
            'data':  {'temperature': 55.5, 'vibration_x': 8.0, 'vibration_y': 7.0,
                      'vibration_z': 9.0, 'vibration_total': 14.0, 'current': 50.0},
            'expect': 'INCERTAIN'
        },
    ]

    rng = np.random.default_rng(42)
    for tc in test_cases:
        d = tc['data']
        # Fenêtre synthétique avec bruit ±3% pour eviter std=0/kurtosis=NaN
        rows = [{k: max(0.0, v * (1 + rng.normal(0, 0.03))) for k, v in d.items()}
                for _ in range(WINDOW_SIZE)]
        window_data = pd.DataFrame(rows, columns=[
            'temperature', 'vibration_x', 'vibration_y',
            'vibration_z', 'vibration_total', 'current'
        ])
        feats = extract_features_from_window(window_data)
        row   = np.array([[feats[f] for f in feature_names]], dtype=np.float32)
        row   = np.nan_to_num(row, nan=0.0)
        row_s = scaler.transform(row)
        row_p = pca.transform(row_s)

        # Vote
        v_if    = 1 if trained['if'].predict(row_p)[0]    == -1 else 0
        v_lof   = 1 if trained['lof'].predict(row_p)[0]   == -1 else 0
        v_ocsvm = 1 if trained['ocsvm'].predict(row_p)[0] == -1 else 0
        ecod_pred = trained['ecod'].predict(row_p)[0]
        if trained.get('ecod_type') == 'pyod':
            v_ecod = 1 if ecod_pred == 1 else 0
        else:
            v_ecod = 1 if ecod_pred == -1 else 0
        total_votes = v_if + v_lof + v_ocsvm + v_ecod
        result = "ANOMALY" if total_votes >= VOTE_THRESHOLD else "NORMAL"  # Vote 2/4
        correct = result == tc['expect'] or tc['expect'] == 'INCERTAIN'
        icon = f"{G}✅{RS}" if correct else f"{Y}⚠ {RS}"
        print(f"\n  {icon} {tc['name']}")
        print(f"     Temp={d['temperature']}°C  VibZ={d['vibration_z']}mg  I={d['current']}A")
        print(f"     Votes : IF={v_if} LOF={v_lof} OCSVM={v_ocsvm} ECOD={v_ecod} → {total_votes}/4")
        print(f"     Résultat : {G if result=='NORMAL' else R}{result}{RS}  (attendu: {tc['expect']})")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — SAUVEGARDE DES MODÈLES ET MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def save_models(trained: dict, scaler, pca, feature_names: list, metrics: dict, n_total: int = 0):
    """
    Sauvegarde tous les artefacts ML dans le dossier models/.

    Fichiers générés :
      model_if_v3.pkl      → Isolation Forest
      model_lof_v3.pkl     → Local Outlier Factor
      model_ocsvm_v3.pkl   → One-Class SVM
      model_ecod_v3.pkl    → ECOD (ou clone IF)
      scaler_v3.pkl        → RobustScaler (médiane + IQR)
      pca_v3.pkl           → PCA (~5 composantes)
      features_v3.pkl      → liste ordonnée des 31 features
      metrics_v3.csv       → métriques F1, AUC, etc.
    """
    head("ÉTAPE 6 — SAUVEGARDE")

    # Seuils optimaux (stockés par evaluate_models pour l'API)
    avg_score     = metrics.pop('_avg_score', None)
    opt_thresholds = metrics.pop('_opt_thresholds', {})
    metrics.pop('_opt_preds', None)
    metrics.pop('_f1_softvote', None)
    if avg_score is not None:
        sv_n   = metrics.get('SoftVote', {}).get('n_anomalies', int(len(avg_score) * CONTAMINATION))
        sv_thr = float(np.percentile(avg_score, 100 - sv_n / len(avg_score) * 100))
    else:
        sv_thr = 0.5
    threshold_data = {
        'softvote_threshold': sv_thr,
        'contamination':      float(CONTAMINATION),
        **opt_thresholds,
    }

    files = {
        'model_if_v3.pkl':      trained['if'],
        'model_lof_v3.pkl':     trained['lof'],
        'model_ocsvm_v3.pkl':   trained['ocsvm'],
        'model_ecod_v3.pkl':    trained['ecod'],
        'scaler_v3.pkl':        scaler,
        'pca_v3.pkl':           pca,
        'features_v3.pkl':      feature_names,
        'threshold_v3.pkl':     threshold_data,
    }

    for fname, obj in files.items():
        path = MODEL_DIR / fname
        joblib.dump(obj, path)
        size_kb = path.stat().st_size / 1024
        ok(f"{fname:30s} → {size_kb:.0f} KB")

    # Métriques CSV — Vote2/4 comme métrique principale (cohérent avec l'API)
    vote_metrics = metrics.get('Vote2/4', metrics.get('SoftVote', {}))
    csv_path = MODEL_DIR / "metrics_v3.csv"
    csv_content = f"""metric,value
f1_score,{vote_metrics.get('f1', 0):.4f}
accuracy,{vote_metrics.get('accuracy', 0):.4f}
precision,{vote_metrics.get('precision', 0):.4f}
recall,{vote_metrics.get('recall', 0):.4f}
auc_roc,{vote_metrics.get('auc_roc', 0):.4f}
n_anomalies,{vote_metrics.get('n_anomalies', 0)}
n_total,{n_total}
contamination,{CONTAMINATION}
model_version,V3
n_features,31
ensemble,IF + LOF + OCSVM + ECOD
voting,Vote2/4 majoritaire
augmentation,x{AUGMENT_FACTOR}
pca_variance,0.95
window_size,{WINDOW_SIZE}
dataset,ai_cp full_data — mesures reelles — 20 capteurs IFM — nov2025-mar2026
ecod_type,{trained.get('ecod_type', 'unknown')}
weights_if,0.25
weights_lof,0.25
weights_ocsvm,0.25
weights_ecod,0.25
trained_at,{datetime.now().isoformat()}
"""
    csv_path.write_text(csv_content, encoding="utf-8")
    ok(f"metrics_v3.csv sauvegardé")

    info(f"Tous les fichiers dans : {MODEL_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════════
# LECTURE MYSQL DIRECTE — full_data (source principale, bonne échelle)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_full_data_from_mysql(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                                password=MYSQL_PASSWORD, database=MYSQL_DATABASE,
                                table=MYSQL_TABLE, sample_n=MYSQL_SAMPLE_N) -> pd.DataFrame:
    """
    Lit full_data depuis MySQL et construit un DataFrame de sessions.
    Même logique de consolidation que realtime_mariadb.py :
      3 lignes (gph=temperature + vibration_x + vibration_y) → 1 session.

    Avantage sur motor_mesure : données à l'échelle production réelle
    (vib_z P50=322 mg, P95=783 mg — vs max 149 mg dans motor_mesure SQL).
    """
    head("ÉTAPE 1 — LECTURE MySQL full_data (source principale)")

    try:
        import mysql.connector
    except ImportError:
        warn("mysql-connector-python non installé → pip install mysql-connector-python")
        return pd.DataFrame()

    try:
        conn = mysql.connector.connect(
            host=host, port=port, user=user, password=password,
            database=database, connection_timeout=10
        )
        cursor = conn.cursor(dictionary=True)
        info(f"Connecté à {host}:{port}/{database}.{table}")
    except Exception as e:
        warn(f"Connexion MySQL échouée : {e}")
        return pd.DataFrame()

    try:
        cursor.execute(f"SELECT MIN(id) as mn, MAX(id) as mx, COUNT(*) as total FROM `{table}`")
        row = cursor.fetchone()
        min_id  = row['mn'] or 0
        max_id  = row['mx'] or 0
        total   = row['total'] or 0

        # Stratégie : prendre les DERNIÈRES lignes (même données que le replay)
        # Le replay rejoue les dernières N lignes → le training doit apprendre la même distribution
        fetch_n = sample_n * 3 * 2   # ×2 pour avoir assez après consolidation
        start_id = max(min_id, max_id - fetch_n)
        info(f"{total:,} lignes dans full_data | training sur les {fetch_n:,} dernières lignes (id > {start_id:,})")

        cursor.execute(f"""
            SELECT id, SensorNodeId, gph, data
            FROM `{table}`
            WHERE id > %s
            ORDER BY id ASC
            LIMIT %s
        """, (start_id, fetch_n))
        rows = cursor.fetchall()
        info(f"{len(rows)} lignes récupérées (période récente)")
    except Exception as e:
        warn(f"Erreur SELECT full_data : {e}")
        conn.close()
        return pd.DataFrame()

    conn.close()

    # Consolidation 3-lignes → session (même logique que realtime_mariadb.py)
    from collections import defaultdict as _dd
    pending  = _dd(dict)
    sessions = []

    for r in rows:
        sensor_id = r.get('SensorNodeId', 'unknown')
        gph       = r.get('gph', '')

        # Parse JSON du champ data (peut être str, bytes, dict ou JSON doublement encodé)
        try:
            raw = r.get('data') or ''
            if isinstance(raw, bytes):
                data = json.loads(raw.decode('utf-8'))
            elif isinstance(raw, dict):
                data = raw
            elif isinstance(raw, str) and raw.strip():
                data = json.loads(raw)
            else:
                data = {}
            # Double-encodage : json.loads peut retourner une str si le JSON est une chaîne quotée
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            continue

        meas_id = data.get('MeasDetails', {}).get('Id') or f"{sensor_id}_{r['id']}"
        key     = f"{sensor_id}_{meas_id}"

        if gph == 'temperature':
            pending[key]['sensor_id']    = sensor_id
            pending[key]['temperature']  = data.get('Temperature')
            vib_rms = data.get('Vibration', {}).get('RMS', {})
            if 'Z' in vib_rms:
                pending[key]['vibration_z'] = vib_rms['Z']
        elif gph == 'vibration_x':
            vib_rms = data.get('Vibration', {}).get('RMS', {})
            if 'X' in vib_rms:
                pending[key]['vibration_x'] = vib_rms['X']
        elif gph == 'vibration_y':
            vib_rms = data.get('Vibration', {}).get('RMS', {})
            if 'Y' in vib_rms:
                pending[key]['vibration_y'] = vib_rms['Y']

        s = pending[key]
        if all(s.get(k) is not None for k in ['temperature', 'vibration_z', 'vibration_x', 'vibration_y']):
            vx = float(s.get('vibration_x', 0) or 0)
            vy = float(s.get('vibration_y', 0) or 0)
            vz = float(s['vibration_z'])
            sessions.append({
                'source':          'full_data',
                'sensor_id':       s.get('sensor_id', sensor_id),
                'temperature':     float(s['temperature']),
                'vibration_x':     vx,
                'vibration_y':     vy,
                'vibration_z':     vz,
                'vibration_total': float(np.sqrt(vx**2 + vy**2 + vz**2)),
                'current':         0.0,
                'timestamp':       datetime.now(),
            })
            del pending[key]
            if len(sessions) >= sample_n:
                break

    if not sessions:
        warn("Aucune session consolidée depuis MySQL — vérifier la structure de full_data")
        return pd.DataFrame()

    df = pd.DataFrame(sessions)
    ok(f"MySQL full_data : {len(df)} sessions | {df['sensor_id'].nunique()} capteurs")
    info(f"Température : {df['temperature'].min():.1f}°C – {df['temperature'].max():.1f}°C")
    info(f"Vib Z       : {df['vibration_z'].min():.1f} – {df['vibration_z'].max():.1f} mg")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import io
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Entraînement des 4 modèles non supervisés")
    parser.add_argument('--sql',      default=DEFAULT_SQL,   help="Chemin vers le fichier .sql")
    parser.add_argument('--db-host',  default=MYSQL_HOST,    help="Hôte MySQL")
    parser.add_argument('--db-user',  default=MYSQL_USER,    help="Utilisateur MySQL")
    parser.add_argument('--db-pass',  default=MYSQL_PASSWORD,help="Mot de passe MySQL")
    parser.add_argument('--db-name',  default=MYSQL_DATABASE,help="Base de données MySQL")
    parser.add_argument('--no-mysql', action='store_true',   help="Forcer utilisation SQL (ignorer MySQL)")
    args = parser.parse_args()

    print(f"\n{B}{C}{'╔'+'═'*60+'╗'}")
    print(f"║  ENTRAÎNEMENT MODÈLES NON SUPERVISÉS V6                   ║")
    print(f"║  Source : MySQL full_data (25 000 sessions réelles)       ║")
    print(f"║  Modèles : IF + LOF + OCSVM + ECOD                        ║")
    print(f"╚{'═'*60+'╝'}{RS}\n")

    t_total = time.time()

    # ── Priorité 1 : MySQL full_data (données production bonne échelle) ──────────
    df_mysql = pd.DataFrame()
    if not args.no_mysql:
        df_mysql = parse_full_data_from_mysql(
            host=args.db_host, user=args.db_user,
            password=args.db_pass, database=args.db_name
        )

    if len(df_mysql) >= 200:
        ok(f"Source MySQL full_data : {len(df_mysql)} sessions à l'échelle production")
        X, feature_names, y_heuristic = build_feature_matrix(df_mysql)

    else:
        # ── Priorité 2 : features production depuis realtime_results.json ────────
        X_prod, feat_names_prod, y_prod = parse_features_from_realtime(REALTIME_RESULTS)
        if len(X_prod) >= 500 and float(y_prod.mean()) >= 0.03:
            ok(f"Source realtime_results.json : {len(X_prod)} vecteurs × 31 features")
            X, feature_names, y_heuristic = X_prod, feat_names_prod, y_prod
            rng = np.random.default_rng(RANDOM_STATE)
            X_aug, y_aug = [X], [y_heuristic]
            for _ in range(AUGMENT_FACTOR - 1):
                noise = rng.normal(0, 0.02, size=X.shape).astype(np.float32)
                X_aug.append(np.clip(X + X * noise, 0, None))
                y_aug.append(y_heuristic)
            X = np.vstack(X_aug)
            y_heuristic = np.concatenate(y_aug)
            X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)
        else:
            # ── Priorité 3 : SQL file (fallback, domain gap) ──────────────────
            warn(f"MySQL indisponible + realtime insuffisant → fallback SQL ({args.sql})")
            df_sql = parse_sql_to_dataframe(args.sql)
            X, feature_names, y_heuristic = build_feature_matrix(df_sql)

    # Recalibrer la contamination (clampé 5%-35%)
    global CONTAMINATION
    CONTAMINATION = round(max(0.05, min(0.35, float(y_heuristic.mean()))), 3)
    info(f"Contamination recalibree : {CONTAMINATION:.1%} ({int(y_heuristic.sum())}/{len(y_heuristic)} sessions anormales)")

    # Étape 3 : Prétraitement
    scaler, pca, X_pca = preprocess(X)

    # Étape 4 : Entraîner les 4 modèles
    trained = train_models(X_pca)

    # Étape 5 : Évaluer
    metrics = evaluate_models(trained, X_pca, y_heuristic, scaler, pca, feature_names)

    # Étape 6 : Sauvegarder
    save_models(trained, scaler, pca, feature_names, metrics, n_total=len(y_heuristic))

    # ── Résumé final ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    head(f"TERMINÉ EN {elapsed:.1f}s")
    ok(f"4 modèles entraînés et sauvegardés dans {MODEL_DIR}/")
    ok(f"Vote2/4 : F1={metrics.get('Vote2/4', {}).get('f1', 0):.4f}  |  Vote3/4 : F1={metrics.get('Vote3/4', {}).get('f1', 0):.4f}")
    ok(f"31 features V6 : +delta_vib, +delta_temp, +vib_entropy, +fft_ratio, +vib_asym_xy, +vib_asym_xz")
    info("Relance l'API : python api_unified_pythagore.py")


if __name__ == "__main__":
    main()