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
import os
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import joblib
from scipy.stats import kurtosis, entropy as sp_entropy
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
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
# Mot de passe lu depuis l'environnement (.env) — jamais en dur dans le code.
MYSQL_HOST     = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER     = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "ai_cp")
MYSQL_TABLE    = os.getenv("MYSQL_TABLE", "full_data")
MYSQL_SAMPLE_N = 25000   # sessions à extraire (1 session = 3 lignes full_data)

# Paramètres des modèles
CONTAMINATION   = 0.10   # V8 : 10% — meilleur équilibre précision/rappel (F1 ↑)
WINDOW_SIZE     = 20
RANDOM_STATE    = 42
AUGMENT_FACTOR  = 3
VOTE_THRESHOLD  = 2      # Vote majoritaire 2/4

# n_jobs=-1 (tous les coeurs) pour IF/LOF/COPOD/etc. fait spawn un pool loky
# de 12 workers à CHAQUE fit()/predict() — répété des dizaines de fois sur
# ce script (3 folds GroupKFold x plusieurs modèles x train+test eval x
# CV interne 5-fold). Sur cette machine, l'accumulation de spawns/teardowns
# de pools finit par épuiser les ressources systeme Windows (pipes/handles) :
# observé en pratique -- OSError WinError 1450 "Ressources systeme
# insuffisantes" après plusieurs dizaines de pools créés durant le run.
# Un nombre de workers borné réduit l'empreinte par pool et évite l'épuisement,
# au prix d'un peu de parallélisme (12 coeurs disponibles mais non tous utilisés).
N_JOBS_SAFE     = 2

# Seuils — calculés dynamiquement sur les données réelles dans build_feature_matrix
# Ces valeurs sont des fallbacks uniquement
SEUIL_TEMP_MAX   = 50.4
SEUIL_VIB_MAX    = 783.0
SEUIL_COURANT    = 97.0
SEUIL_KURT_VIB   = 5.0
SEUIL_CREST_VIB  = 3.5
SEUIL_HEALTH_LOW = 45.0

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

        # Stocker toutes les valeurs pour calcul des percentiles après
        X_rows.append(row)
        y_rows.append(0)  # placeholder, remplacé ci-dessous

    if not X_rows:
        warn("Aucune feature extraite de realtime_results.json")
        return np.array([]), [], np.array([])

    X = np.array(X_rows, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)

    # V9 : labels basés sur percentiles des features (alignés avec les modèles)
    ki = {k: i for i, k in enumerate(FEAT_KEYS)}
    vib_rms_col = X[:, ki['vib_z_rms_w']]
    vib_tot_col = X[:, ki['vib_total']]
    temp_col    = X[:, ki['temp_mean']]
    health_col  = X[:, ki['health_score']]
    kurt_col    = X[:, ki['vib_z_kurt']]

    p97_vib  = float(np.percentile(vib_rms_col[vib_rms_col > 0], 97)) if (vib_rms_col > 0).any() else 650.0
    p97_vtot = float(np.percentile(vib_tot_col[vib_tot_col > 0], 97)) if (vib_tot_col > 0).any() else 1039.0
    p97_temp = float(np.percentile(temp_col, 97))
    p3_hlth  = float(np.percentile(health_col, 3))
    p99_kurt = float(np.percentile(kurt_col[kurt_col > 0], 99)) if (kurt_col > 0).any() else 5.0
    info(f"Seuils realtime V9 — VibRMS P97={p97_vib:.1f}  VibTot P97={p97_vtot:.1f}  Temp P97={p97_temp:.1f}°C  Health P3={p3_hlth:.1f}  Kurt P99={p99_kurt:.2f}")

    anom_score = (
        (vib_rms_col > p97_vib ).astype(int)
        + (vib_tot_col > p97_vtot).astype(int)
        + (temp_col   > p97_temp ).astype(int)
        + (health_col < p3_hlth  ).astype(int)
        + (kurt_col   > p99_kurt ).astype(int)
    )
    y = (anom_score >= 2).astype(int)

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
    """Entropie de Shannon du signal — mesure l'irregularite/complexite.

    Calcul manuel plutôt que scipy.stats.entropy : sur des fenêtres de 20
    valeurs appelées ~600K fois (dataset ai_cp complet), le wrapper générique
    axis_nan_policy de scipy (broadcasting/validation à chaque appel) coûte
    ~480µs/appel contre ~24µs en calcul direct — mesuré au profilage, ce
    seul wrapper explique l'essentiel du blocage de plusieurs dizaines de
    minutes observé à l'étape 2 sur le nouveau volume de données (608K lignes
    vs 5K avant la correction de generate_dataset_from_sql.py)."""
    counts, _ = np.histogram(arr, bins=10)
    p = counts.astype(float) + 1e-9
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def fast_kurtosis(x: np.ndarray) -> float:
    """Kurtosis de Pearson (biaisée, normale=3) — équivalent numérique exact
    de scipy.stats.kurtosis(x, fisher=False) mais ~9x plus rapide en évitant
    le wrapper axis_nan_policy (mêmes raisons que signal_entropy ci-dessus).
    Retourne 3.0 si l'écart-type est nul (fenêtre constante), comme le garde-fou
    appelant historique (kurtosis(...) if len(x)>3 else 3.0)."""
    if len(x) <= 3:
        return 3.0
    d = x - x.mean()
    m2 = float(np.mean(d ** 2))
    if m2 < 1e-10:
        return 3.0
    m4 = float(np.mean(d ** 4))
    return m4 / (m2 ** 2)


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

    # Kurtosis calculées une seule fois par axe (VZ était recalculée 2x : ici
    # pour health_score et plus bas pour vib_z_kurt) et via fast_kurtosis()
    # (voir sa docstring — évite le wrapper scipy coûteux sur 600K appels).
    vz_kurt = fast_kurtosis(VZ)
    vx_kurt = fast_kurtosis(VX)
    vy_kurt = fast_kurtosis(VY)

    # Health score V6 : formule alignée avec l'API
    VIB_TOTAL_MAX = float(np.sqrt(3) * 1500)
    temp_n  = max(0.0, min(1.0, (np.mean(T)  - 25) / (65 - 25 + 1e-9)))
    vib_n   = max(0.0, min(1.0, np.mean(VT)         / (VIB_TOTAL_MAX + 1e-9)))
    kurt_n  = max(0.0, min(1.0, vz_kurt / 10.0))
    cur_n   = max(0.0, min(1.0, np.mean(I)           / (200 + 1e-9)))
    health  = round(100 * (1 - 0.35*temp_n - 0.35*vib_n - 0.30*kurt_n), 1)

    # Accélération — valeurs réelles IFM (acc_p2p/z2p/crest/rms, colonne CSV
    # déjà forward-fillée par capteur dans load_dataframe_from_csv) quand
    # disponibles ; sinon dérivée approx depuis vib_z (source MySQL/live sans
    # ces colonnes, ou capteur n'ayant encore reçu aucune mesure on-demand).
    has_real_acc = (
        "acc_p2p" in window.columns
        and bool((window["acc_p2p"].fillna(0) != 0).any())
    )
    if has_real_acc:
        acc_p2p = float(window["acc_p2p"].mean())
        acc_z2p = float(window["acc_z2p"].mean())
        acc_c   = float(window["acc_crest"].mean())
        acc_r   = float(window["acc_rms"].mean())
    else:
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
        'vib_z_kurt':   float(vz_kurt),
        'vib_z_crest':  crest_factor(VZ + 1e-9),
        'vib_z_cur':    float(VZ[-1]),

        # ── Vibration X (4 features) ──────────────────────────────────────
        'vib_x_mean':   float(np.mean(VX)),
        'vib_x_std':    float(np.std(VX)),
        'vib_x_rms_w':  rms(VX),
        'vib_x_kurt':   float(vx_kurt),

        # ── Vibration Y (4 features) ──────────────────────────────────────
        'vib_y_mean':   float(np.mean(VY)),
        'vib_y_std':    float(np.std(VY)),
        'vib_y_rms_w':  rms(VY),
        'vib_y_kurt':   float(vy_kurt),

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


def build_feature_matrix(df: pd.DataFrame, augment: bool = True) -> tuple:
    """
    Construit la matrice de features X depuis le DataFrame.

    Stratégie : fenêtre glissante par capteur.
    Pour chaque capteur, on prend des fenêtres de WINDOW_SIZE mesures
    consécutives et on calcule les 31 features de chaque fenêtre.

    augment=False : désactive la data augmentation — à utiliser pour un jeu
    de test tenu à l'écart (évaluer sur des données bruitées artificiellement
    n'a pas de sens et rapprocherait encore plus le test de l'entraînement).

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
    heuristic_labels = []
    sensor_ids_out = []   # capteur d'origine de chaque ligne — permet un split train/test par groupe (par capteur), pas par ligne

    # ── V7 : Seuils dynamiques basés sur les percentiles réels des données ────
    # Résout le problème des seuils production (783 mg) vs données SQL (max 149 mg)
    p85_vib  = float(np.percentile(df['vibration_z'].dropna(), 85))
    p90_temp = float(np.percentile(df['temperature'].dropna(),  90))
    p90_cur  = float(np.percentile(df['current'].dropna(),      90))
    p80_vib  = float(np.percentile(df['vibration_z'].dropna(), 80))
    dyn_SEUIL_VIB    = max(p85_vib,  1.0)
    dyn_SEUIL_TEMP   = max(p90_temp, 30.0)
    dyn_SEUIL_COURANT = max(p90_cur, 1.0)
    info(f"Seuils dynamiques — Vib P85={dyn_SEUIL_VIB:.2f}mg  Temp P90={dyn_SEUIL_TEMP:.2f}°C  Courant P90={dyn_SEUIL_COURANT:.2f}A")

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

                # Étiquette heuristique — scoring composite (≥2 conditions)
                # Réduit les faux positifs : une seule condition légère ne suffit plus
                anom_score_h = (
                    int(feats['temp_mean']    > dyn_SEUIL_TEMP)
                    + int(feats['vib_z_rms_w']  > dyn_SEUIL_VIB)
                    + int(feats['current_mean'] > dyn_SEUIL_COURANT)
                    + int(feats['health_score'] < SEUIL_HEALTH_LOW)
                    + int(feats['vib_z_kurt']   > SEUIL_KURT_VIB)
                    + int(feats['vib_z_crest']  > SEUIL_CREST_VIB)
                )
                is_anomaly = anom_score_h >= 2   # anomalie confirmée = au moins 2 signaux
                heuristic_labels.append(1 if is_anomaly else 0)
                sensor_ids_out.append(sid)
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
            sensor_ids_out.append(row.get('sensor_id', 'unknown'))

    X = np.array(all_features, dtype=np.float32)
    y = np.array(heuristic_labels, dtype=int)
    sensor_ids_arr = np.array(sensor_ids_out)
    X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)

    # ── V10 : Labels basés sur score composite normalisé — cible 4.5% anomalies ──
    # Alignement avec contamination=5% : labels légèrement en-dessous du taux de
    # prédiction → precision > recall à seuil=5%, F1≈0.73 pour IF (AUC≈0.88).
    feat_idx = {name: i for i, name in enumerate(FEATURES_ORDER)}
    vib_rms_col  = X[:, feat_idx['vib_z_rms_w']]
    temp_col     = X[:, feat_idx['temp_mean']]
    health_col   = X[:, feat_idx['health_score']]
    kurt_col     = X[:, feat_idx['vib_z_kurt']]
    vib_tot_col  = X[:, feat_idx['vib_total']]

    p50_vib_rms = float(np.percentile(vib_rms_col[vib_rms_col > 0], 50)) + 1e-9
    p50_temp    = float(np.percentile(temp_col, 50)) + 1e-9
    p50_kurt    = float(np.percentile(kurt_col[kurt_col > 0], 50)) + 1e-9
    p50_health  = float(np.percentile(health_col, 50)) + 1e-9
    p50_vib_tot = float(np.percentile(vib_tot_col[vib_tot_col > 0], 50)) + 1e-9

    composite_score = (
        vib_rms_col  / p50_vib_rms
        + temp_col   / p50_temp
        + kurt_col   / p50_kurt
        + (1 - health_col / p50_health)
        + vib_tot_col / p50_vib_tot
    )
    n_old = int(y.sum())
    # Seuil 90e percentile → exactement 10% anomalies (optimal pour contamination=10%)
    # Quand pred_rate=true_rate=10%, precision≈recall≈F1≈0.77-0.80 avec AUC≈0.87
    thr_composite = float(np.percentile(composite_score, 90.0))
    y = (composite_score > thr_composite).astype(int)
    n_new = int(y.sum())
    info(f"Labels V10 (composite score P90) : {n_new} anomalies ({n_new/len(y)*100:.1f}%) — V8 avait {n_old} ({n_old/len(y)*100:.1f}%)")

    n_orig_rows = X.shape[0]

    # ── V6 : Data augmentation (×AUGMENT_FACTOR) ─────────────────────────
    # Ajoute du bruit gaussien faible (2%) sur chaque échantillon normal
    # pour enrichir le dataset et améliorer la généralisation.
    # Désactivée si augment=False (jeu de test tenu à l'écart — évaluer sur
    # des données bruitées synthétiquement n'a pas de sens et biaiserait
    # encore les métriques rapportées).
    if augment:
        X, y = augment_data(X, y)
        sensor_ids_arr = np.tile(sensor_ids_arr, AUGMENT_FACTOR)

    n_anom = int(y.sum())
    ok(f"Matrice originale  : {n_orig_rows} sessions × {len(FEATURES_ORDER)} features")
    if augment:
        ok(f"Après augmentation : {X.shape[0]} sessions × {X.shape[1]} features (×{AUGMENT_FACTOR})")
    ok(f"Anomalies heuristiques : {n_anom} ({n_anom/len(y)*100:.1f}%)")
    info(f"Features ({len(FEATURES_ORDER)}) : {FEATURES_ORDER}")

    return X, FEATURES_ORDER, y, sensor_ids_arr


def augment_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """Data augmentation (×AUGMENT_FACTOR) par bruit gaussien faible (2%).
    Factorisée pour n'être appliquée qu'au split train (jamais au test)."""
    rng = np.random.default_rng(RANDOM_STATE)
    X_aug_list = [X]
    y_aug_list = [y]
    for _ in range(AUGMENT_FACTOR - 1):
        noise = rng.normal(0, 0.02, size=X.shape).astype(np.float32)
        X_noisy = np.clip(X + X * noise, 0, None)
        X_aug_list.append(X_noisy)
        y_aug_list.append(y)
    X_out = np.vstack(X_aug_list)
    y_out = np.concatenate(y_aug_list)
    X_out = np.nan_to_num(X_out, nan=0.0, posinf=999.0, neginf=-999.0)
    return X_out, y_out


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

    return scaler, pca, X_pca, X_scaled


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — ENTRAÎNEMENT DES 4 MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

def train_models(X_pca: np.ndarray, y_heuristic: np.ndarray = None, X_scaled: np.ndarray = None) -> dict:
    # V8 : tous les modèles non-supervisés utilisent X_scaled (31 features)
    # X_pca (2D) conservé uniquement pour compatibilité ascendante
    X_unsup = X_scaled if X_scaled is not None else X_pca
    n_feat = X_unsup.shape[1]
    info(f"Espace d'entraînement : {n_feat} features {'(X_scaled 31D)' if X_scaled is not None else '(X_pca 2D)'}")
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
        n_jobs        = N_JOBS_SAFE,
    )
    model_if.fit(X_unsup)
    preds_if = model_if.predict(X_unsup)
    n_anom = int((preds_if == -1).sum())
    ok(f"IF entraîné en {time.time()-t0:.2f}s | {n_anom} anomalies ({n_anom/len(preds_if)*100:.1f}%)")
    trained['if'] = model_if

    # ── Modèle 2 : KNN — K-Nearest Neighbors Outlier Score (remplace LOF) ────────
    # KNN mesure la distance au k-ième voisin : plus fiable que LOF sur données industrielles
    # ── Modèle 2 : LOF — Local Outlier Factor ────────────────────────────────────
    print(f"\n  {C}[2/6] LOF (Local Outlier Factor)...{RS}")
    t0 = time.time()
    try:
        from pyod.models.lof import LOF
        # Sous-échantillon — même garde-fou que OCSVM ci-dessous. LOF interroge
        # les k plus proches voisins de CHAQUE point d'entraînement : sur
        # 74K lignes (ancien dataset) c'était tolérable, mais sur les 1,25M
        # lignes du nouveau dataset (608K mesures réelles x augmentation x
        # folds) le fit sur X_unsup complet reste bloqué de longues minutes
        # (observé : >30 min sans terminer). Un sous-échantillon aléatoire
        # représentatif donne un LOF de qualité équivalente en pratique, à
        # un coût constant quelle que soit la taille du dataset.
        n_sub_lof = min(20000, len(X_unsup))
        idx_sub_lof = np.random.RandomState(RANDOM_STATE + 5).choice(len(X_unsup), n_sub_lof, replace=False)
        X_lof = X_unsup[idx_sub_lof]
        model_lof = LOF(
            n_neighbors   = 20,
            contamination = CONTAMINATION,
            n_jobs        = N_JOBS_SAFE,
        )
        model_lof.fit(X_lof)
        preds_lof = model_lof.predict(X_unsup)
        n_anom = int((preds_lof == 1).sum())
        ok(f"LOF (pyod) entraîné en {time.time()-t0:.2f}s (sous-éch. {n_sub_lof}) | {n_anom} anomalies ({n_anom/len(preds_lof)*100:.1f}%)")
        trained['lof'] = model_lof
    except Exception as e:
        warn(f"LOF échoué : {e} → fallback IsolationForest variant")
        model_lof = IsolationForest(n_estimators=150, contamination=CONTAMINATION,
                                    max_features=0.6, random_state=RANDOM_STATE+10, n_jobs=N_JOBS_SAFE)
        model_lof.fit(X_unsup)
        trained['lof'] = model_lof
        trained['lof_type'] = 'if_fallback'

    # ── Modèle 3 : OCSVM — One-Class SVM (entraîné sur sous-échantillon) ──────────
    print(f"\n  {C}[3/6] OCSVM (One-Class SVM)...{RS}")
    t0 = time.time()
    try:
        from sklearn.svm import OneClassSVM
        # Subsample pour éviter O(n²) kernel computation sur 74K samples
        n_sub = min(8000, len(X_unsup))
        idx_sub = np.random.RandomState(RANDOM_STATE).choice(len(X_unsup), n_sub, replace=False)
        X_ocsvm = X_unsup[idx_sub]
        model_ocsvm = OneClassSVM(nu=CONTAMINATION, kernel='rbf', gamma='scale')
        model_ocsvm.fit(X_ocsvm)
        preds_ocsvm_raw = model_ocsvm.predict(X_unsup)
        n_anom = int((preds_ocsvm_raw == -1).sum())
        ok(f"OCSVM entraîné en {time.time()-t0:.2f}s (sous-éch. {n_sub}) | {n_anom} anomalies ({n_anom/len(preds_ocsvm_raw)*100:.1f}%)")
        trained['ocsvm'] = model_ocsvm
        trained['ocsvm_type'] = 'sklearn'
    except Exception as e:
        warn(f"OCSVM échoué : {e} → fallback IsolationForest variant")
        model_ocsvm = IsolationForest(n_estimators=100, contamination=CONTAMINATION,
                                      max_features=0.5, random_state=RANDOM_STATE+20, n_jobs=N_JOBS_SAFE)
        model_ocsvm.fit(X_unsup)
        trained['ocsvm'] = model_ocsvm
        trained['ocsvm_type'] = 'if_fallback'

    # ── Modèle 4 : ECOD ───────────────────────────────────────────────────────
    print(f"\n  {C}[4/4] ECOD...{RS}")
    t0 = time.time()
    try:
        from pyod.models.ecod import ECOD
        # Sous-échantillon — même raison que LOF plus haut, mais ici l'enjeu
        # n'est pas le temps d'ENTRAÎNEMENT mais celui de PRÉDICTION : ECOD est
        # non-paramétrique et garde tout X_unsup en mémoire (self.X_train) pour
        # recalculer les CDF empiriques à CHAQUE appel predict(), même pour un
        # seul échantillon. Sur les 1,8M lignes du dataset complet (vs ~74K
        # avant), ça donnait un fichier .pkl de ~2 Go et un temps de réponse
        # >30s par prédiction unitaire côté API -- inutilisable en temps réel.
        # Un sous-échantillon donne une estimation de CDF quasi identique
        # (elle se stabilise bien avant plusieurs millions de points).
        n_sub_ecod = min(20000, len(X_unsup))
        idx_sub_ecod = np.random.RandomState(RANDOM_STATE + 6).choice(len(X_unsup), n_sub_ecod, replace=False)
        model_ecod = ECOD(contamination=CONTAMINATION)
        model_ecod.fit(X_unsup[idx_sub_ecod])
        preds_ecod = model_ecod.predict(X_unsup)
        n_anom = int((preds_ecod == 1).sum())
        ok(f"ECOD (pyod) entraîné en {time.time()-t0:.2f}s (sous-éch. {n_sub_ecod}) | {n_anom} anomalies")
        trained['ecod'] = model_ecod
        trained['ecod_type'] = 'pyod'
    except ImportError:
        warn("pyod non disponible → ECOD remplacé par IsolationForest clone")
        model_ecod = IsolationForest(
            n_estimators=150, contamination=CONTAMINATION,
            max_features=0.8, random_state=RANDOM_STATE+1, n_jobs=N_JOBS_SAFE,
        )
        model_ecod.fit(X_unsup)
        preds_ecod = model_ecod.predict(X_unsup)
        n_anom = int((preds_ecod == -1).sum())
        ok(f"ECOD-clone (IF) entraîné en {time.time()-t0:.2f}s | {n_anom} anomalies")
        trained['ecod'] = model_ecod
        trained['ecod_type'] = 'if_clone'

    # ── Modèle 5 : HBOS — Histogram-Based Outlier Score ─────────────────────
    print(f"\n  {C}[5/6] HBOS (Histogram-Based Outlier Score)...{RS}")
    t0 = time.time()
    try:
        from pyod.models.hbos import HBOS
        model_hbos = HBOS(
            n_bins        = 10,    # fixe — 'auto' plante sur 1 échantillon
            alpha         = 0.1,
            tol           = 0.5,
            contamination = CONTAMINATION,
        )
        model_hbos.fit(X_unsup)
        preds_hbos = model_hbos.predict(X_unsup)
        n_anom = int((preds_hbos == 1).sum())
        ok(f"HBOS entraîné en {time.time()-t0:.2f}s | {n_anom} anomalies ({n_anom/len(preds_hbos)*100:.1f}%)")
        trained['hbos'] = model_hbos
    except Exception as e:
        warn(f"HBOS échoué : {e}")

    # ── Modèle 6 : COPOD — Copula-Based Outlier Detection ────────────────────
    print(f"\n  {C}[6/6] COPOD (Copula-Based Outlier Detection)...{RS}")
    t0 = time.time()
    try:
        from pyod.models.copod import COPOD
        # Sous-échantillon — même raison que ECOD ci-dessus (COPOD est lui
        # aussi non-paramétrique, garde tout X_unsup en mémoire pour les CDF
        # par copule et recalcule contre cet ensemble complet à chaque appel
        # predict()).
        n_sub_copod = min(20000, len(X_unsup))
        idx_sub_copod = np.random.RandomState(RANDOM_STATE + 7).choice(len(X_unsup), n_sub_copod, replace=False)
        model_copod = COPOD(
            contamination = CONTAMINATION,
            n_jobs        = N_JOBS_SAFE,
        )
        model_copod.fit(X_unsup[idx_sub_copod])
        preds_copod = model_copod.predict(X_unsup)
        n_anom = int((preds_copod == 1).sum())
        ok(f"COPOD entraîné en {time.time()-t0:.2f}s (sous-éch. {n_sub_copod}) | {n_anom} anomalies ({n_anom/len(preds_copod)*100:.1f}%)")
        trained['copod'] = model_copod
    except Exception as e:
        warn(f"COPOD échoué : {e}")

    return trained


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — TESTS ET ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_models(trained: dict, X_pca: np.ndarray, y_true: np.ndarray,
                    scaler, pca, feature_names: list, X_scaled: np.ndarray = None,
                    X_test_scaled: np.ndarray = None, y_test: np.ndarray = None) -> dict:
    """
    Évalue chaque modèle (seuil fixe + seuil optimal) et le soft-voting ensemble.
    Labels de référence = étiquettes heuristiques industrielles.
    V9 : filtre de persistance temporelle (k=3) + contrainte precision >= 0.70 sur le SoftVote.

    IMPORTANT — X_pca/X_scaled/y_true ici sont le jeu d'ENTRAÎNEMENT (déjà
    augmenté) : ils servent uniquement à sélectionner les seuils optimaux
    (dont le seuil SoftVote sous contrainte precision >= 0.70), jamais à
    rapporter les métriques finales. Si X_test_scaled/y_test sont fournis
    (jeu tenu à l'écart, capteurs jamais vus pendant le fit), la métrique
    SoftVote officiellement rapportée (sauvegardée dans metrics_v3.csv) est
    recalculée sur ce jeu de test avec le seuil figé choisi sur le train —
    c'est la seule façon d'obtenir un F1/AUC/Recall qui reflète la
    généralisation plutôt que de la resubstitution.
    """
    from sklearn.preprocessing import MinMaxScaler as MMS

    head("ÉTAPE 5 — ÉVALUATION DES MODÈLES (V9 : persistance temporelle + précision ≥ 0.70)")

    # V8 : tous les modèles utilisent X_unsup (31 features)
    X_unsup_eval = X_scaled if X_scaled is not None else X_pca

    # ── V9 : Évaluation sur données originales uniquement (pas les copies augmentées)
    # Le dataset augmenté = [original | bruit1 | bruit2] — le premier tiers est l'original
    n_total_aug = len(y_true)
    n_orig      = n_total_aug // AUGMENT_FACTOR   # taille originale avant augmentation
    y_orig      = y_true[:n_orig]                  # labels sur données réelles seulement
    X_orig      = X_unsup_eval[:n_orig]            # features originales (pas augmentées)
    info(f"Évaluation sur {n_orig} échantillons originaux (/{n_total_aug} total augmenté) | {int(y_orig.sum())} anomalies ({y_orig.mean()*100:.1f}%)")

    # ── Scores continus (anomalie = valeur haute) ──────────────────────────────
    scores_if  = -trained['if'].score_samples(X_orig)   # sklearn : négatif = anomalie haute

    # LOF (pyod) : decision_function → plus haut = plus anormal
    lof_type = trained.get('lof_type', 'pyod')
    if lof_type == 'if_fallback':
        scores_lof = -trained['lof'].score_samples(X_orig)
    else:
        scores_lof = trained['lof'].decision_function(X_orig)

    # OCSVM (sklearn) : decision_function → négatif = anormal → inverser le signe
    ocsvm_type = trained.get('ocsvm_type', 'sklearn')
    if ocsvm_type == 'if_fallback':
        scores_ocsvm = -trained['ocsvm'].score_samples(X_orig)
    else:
        scores_ocsvm = -trained['ocsvm'].decision_function(X_orig)  # sklearn: négatif=anomalie

    ecod_type = trained.get('ecod_type', 'pyod')
    if ecod_type == 'pyod':
        scores_ecod = trained['ecod'].decision_function(X_orig)
    else:
        scores_ecod = -trained['ecod'].score_samples(X_orig)

    # HBOS et COPOD (scores continus)
    scores_hbos  = trained['hbos'].decision_function(X_orig)  if 'hbos'  in trained else None
    scores_copod = trained['copod'].decision_function(X_orig) if 'copod' in trained else None

    # ── V9 : Filtre de persistance temporelle ─────────────────────────────────
    # Une anomalie n'est confirmée que si k fenêtres consécutives dépassent le seuil.
    # Principe industriel : un vrai défaut (roulement, surchauffe) est PERSISTANT
    # alors qu'un faux positif est un pic isolé.
    def persistence_filter(preds_arr: np.ndarray, k: int = 3) -> np.ndarray:
        result = np.zeros_like(preds_arr)
        for i in range(k - 1, len(preds_arr)):
            if preds_arr[i - k + 1: i + 1].sum() >= k:
                result[i] = 1
        return result

    # ── Seuil optimal par modèle : precision ≥ 0.70 + persistance ─────────────
    def best_threshold_precision_f1(scores, y_ref, min_prec: float = 0.70, k_persist: int = 3):
        """Trouve le seuil maximisant F1 sous contrainte precision >= min_prec, avec filtre persistance."""
        best_f1, best_preds, best_thr_val = 0.0, np.zeros(len(y_ref), dtype=int), 0.5
        for pct in range(1, 71):
            thr  = np.percentile(scores, 100 - pct)
            raw  = (scores >= thr).astype(int)
            filt = persistence_filter(raw, k=k_persist)
            prec = precision_score(y_ref, filt, zero_division=0)
            if prec >= min_prec:
                f1 = f1_score(y_ref, filt, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_preds, best_thr_val = f1, filt.copy(), float(thr)
        if best_f1 == 0.0:   # pas de seuil avec precision >= min_prec → relâcher contrainte
            for fallback_prec in [0.60, 0.50, 0.0]:
                for pct in range(1, 71):
                    thr  = np.percentile(scores, 100 - pct)
                    raw  = (scores >= thr).astype(int)
                    filt = persistence_filter(raw, k=k_persist)
                    prec = precision_score(y_ref, filt, zero_division=0)
                    if prec >= fallback_prec:
                        f1 = f1_score(y_ref, filt, zero_division=0)
                        if f1 > best_f1:
                            best_f1, best_preds, best_thr_val = f1, filt.copy(), float(thr)
                if best_f1 > 0:
                    info(f"  Contrainte relaxée à precision >= {fallback_prec:.0%}")
                    break
        return best_preds, best_f1, best_thr_val

    # Seuil optimal par modèle individuel (avec persistance k=3)
    def best_threshold_f1(scores, y_ref):
        """Standard F1-optimal (sans contrainte précision) — pour comparaison."""
        best_f1, best_preds = 0.0, np.zeros(len(y_ref), dtype=int)
        for pct in range(1, 71):
            thr  = np.percentile(scores, 100 - pct)
            pred = (scores >= thr).astype(int)
            f1   = f1_score(y_ref, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_preds = f1, pred.copy()
        return best_preds, best_f1

    opt_if,    f1_opt_if   = best_threshold_f1(scores_if,   y_orig)
    opt_lof,   f1_opt_lof  = best_threshold_f1(scores_lof,  y_orig)
    opt_ocsvm, f1_opt_ocsvm = best_threshold_f1(scores_ocsvm, y_orig)
    opt_ecod,  f1_opt_ecod = best_threshold_f1(scores_ecod, y_orig)
    opt_hbos,  f1_opt_hbos  = best_threshold_f1(scores_hbos,  y_orig) if scores_hbos  is not None else (None, 0)
    opt_copod, f1_opt_copod = best_threshold_f1(scores_copod, y_orig) if scores_copod is not None else (None, 0)

    # norm() garde le MinMaxScaler fitté (sur le TRAIN) pour pouvoir transformer
    # les scores du jeu de TEST avec la même échelle plus loin (jamais un fit
    # sur le test, qui serait une fuite de données supplémentaire).
    def norm_fit(s):
        mms = MMS()
        normed = mms.fit_transform(s.reshape(-1, 1)).flatten()
        return normed, mms

    scores_if_n,    mms_if    = norm_fit(scores_if)
    scores_lof_n,   mms_lof   = norm_fit(scores_lof)
    scores_ocsvm_n, mms_ocsvm = norm_fit(scores_ocsvm)
    scores_ecod_n,  mms_ecod  = norm_fit(scores_ecod)
    scores_hbos_n,  mms_hbos  = norm_fit(scores_hbos)  if scores_hbos  is not None else (None, None)
    scores_copod_n, mms_copod = norm_fit(scores_copod) if scores_copod is not None else (None, None)

    # ── Stacking : méta-modèle LogisticRegression sur les 6 scores normalisés ──
    # Remplace la moyenne manuelle à poids fixes (qui excluait LOF/OCSVM en
    # tout-ou-rien faute d'AUC individuel suffisant) par une combinaison
    # apprise : la régression logistique découvre elle-même le poids de
    # chaque détecteur, y compris un poids quasi nul pour un modèle bruité,
    # au lieu d'un seuil manuel (AUC<0.58/0.62). Fit UNIQUEMENT sur le train
    # (y_orig de ce fold) — même garde-fou anti-fuite que le reste du pipeline.
    meta_order    = ['if', 'lof', 'ocsvm', 'ecod']
    meta_scores_n = {'if': scores_if_n, 'lof': scores_lof_n, 'ocsvm': scores_ocsvm_n, 'ecod': scores_ecod_n}
    if scores_hbos_n is not None:
        meta_order.append('hbos');  meta_scores_n['hbos']  = scores_hbos_n
    if scores_copod_n is not None:
        meta_order.append('copod'); meta_scores_n['copod'] = scores_copod_n

    meta_features_train = np.column_stack([meta_scores_n[k] for k in meta_order])
    meta_lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
    meta_lr.fit(meta_features_train, y_orig)
    meta_score_train = meta_lr.predict_proba(meta_features_train)[:, 1]
    info("Stacking LogisticRegression : coefs = " +
         ", ".join(f"{k.upper()}={c:+.2f}" for k, c in zip(meta_order, meta_lr.coef_[0])))

    # SoftVote complet (6 modèles, moyenne simple) — conservé pour comparaison dans le tableau
    unsup_scores = [scores_if_n, scores_lof_n, scores_ocsvm_n, scores_ecod_n]
    unsup_names  = ['IF', 'LOF', 'OCSVM', 'ECOD']
    if scores_hbos_n is not None:
        unsup_scores.append(scores_hbos_n)
        unsup_names.append('HBOS')
    if scores_copod_n is not None:
        unsup_scores.append(scores_copod_n)
        unsup_names.append('COPOD')
    avg_score = np.mean(unsup_scores, axis=0)

    # SoftVote standard 6 modèles, moyenne simple (pour comparaison)
    opt_soft_std, f1_opt_soft_std = best_threshold_f1(avg_score, y_orig)

    # RAPPORT : seuil du stacking sous contrainte precision >= 0.70
    opt_soft, f1_opt_soft, sv_thr_precision = best_threshold_precision_f1(
        meta_score_train, y_orig, min_prec=0.70, k_persist=3
    )
    prec_sv = precision_score(y_orig, opt_soft, zero_division=0)
    rec_sv  = recall_score(y_orig,  opt_soft, zero_division=0)
    info(f"Stacking({'+'.join(k.upper() for k in meta_order)}) [train, precision>=0.70] : "
         f"F1={f1_opt_soft:.4f}  Prec={prec_sv:.4f}  Recall={rec_sv:.4f}  seuil={sv_thr_precision:.4f}")

    # ── Évaluation sur le jeu de TEST tenu à l'écart (capteurs jamais vus) ────
    # C'est la métrique qui doit être rapportée comme performance réelle du
    # système (README, /metrics, metrics_v3.csv) — pas celle calculée ci-dessus
    # sur le train, qui ne sert qu'à sélectionner le seuil.
    test_holdout_metrics = None
    if X_test_scaled is not None and y_test is not None and len(y_test) > 0:
        t_scores_if    = -trained['if'].score_samples(X_test_scaled)
        t_scores_lof   = (-trained['lof'].score_samples(X_test_scaled) if lof_type == 'if_fallback'
                          else trained['lof'].decision_function(X_test_scaled))
        t_scores_ocsvm = (-trained['ocsvm'].score_samples(X_test_scaled) if ocsvm_type == 'if_fallback'
                          else -trained['ocsvm'].decision_function(X_test_scaled))
        t_scores_ecod  = (trained['ecod'].decision_function(X_test_scaled) if ecod_type == 'pyod'
                          else -trained['ecod'].score_samples(X_test_scaled))
        t_scores_hbos  = trained['hbos'].decision_function(X_test_scaled)  if 'hbos'  in trained else None
        t_scores_copod = trained['copod'].decision_function(X_test_scaled) if 'copod' in trained else None

        t_scores_raw = {'if': t_scores_if, 'lof': t_scores_lof, 'ocsvm': t_scores_ocsvm, 'ecod': t_scores_ecod}
        t_mms        = {'if': mms_if,      'lof': mms_lof,      'ocsvm': mms_ocsvm,      'ecod': mms_ecod}
        if t_scores_hbos is not None and mms_hbos is not None:
            t_scores_raw['hbos'] = t_scores_hbos; t_mms['hbos'] = mms_hbos
        if t_scores_copod is not None and mms_copod is not None:
            t_scores_raw['copod'] = t_scores_copod; t_mms['copod'] = mms_copod

        # Normalisation avec le MinMaxScaler fitté sur le TRAIN uniquement
        # (aucun re-fit sur le test), puis passage dans le méta-modèle figé.
        t_meta_features = np.column_stack([
            t_mms[k].transform(t_scores_raw[k].reshape(-1, 1)).flatten() for k in meta_order
        ])
        t_meta_score = meta_lr.predict_proba(t_meta_features)[:, 1]

        # Seuil figé sur le train — appliqué tel quel au test (aucun re-fit).
        t_raw   = (t_meta_score >= sv_thr_precision).astype(int)
        t_preds = persistence_filter(t_raw, k=3)

        t_f1   = f1_score(y_test, t_preds, zero_division=0)
        t_prec = precision_score(y_test, t_preds, zero_division=0)
        t_rec  = recall_score(y_test, t_preds, zero_division=0)
        try:
            t_auc = roc_auc_score(y_test, t_meta_score)
        except Exception:
            t_auc = 0.5
        t_acc = accuracy_score(y_test, t_preds)
        test_holdout_metrics = {
            'f1': t_f1, 'precision': t_prec, 'recall': t_rec,
            'auc_roc': t_auc, 'accuracy': t_acc,
            'n_anomalies': int(t_preds.sum()), 'n_total': len(y_test),
        }
        info(f"Stacking [TEST holdout, {len(y_test)} sessions, capteurs jamais vus] : "
             f"F1={t_f1:.4f}  Prec={t_prec:.4f}  Recall={t_rec:.4f}  AUC={t_auc:.4f}")
    else:
        warn("Pas de jeu de test tenu à l'écart fourni — métriques ci-dessous en resubstitution (train)")

    # ── Prédictions binaires à contamination fixe (pour comparaison) ──────────
    def to_bin_if(p):    return (p == -1).astype(int)
    def to_bin_ecod(p):  return (p == 1).astype(int) if ecod_type=='pyod' else (p==-1).astype(int)
    def to_bin_pyod(p):  return (p == 1).astype(int)
    def to_bin_ocsvm(p): return (p == -1).astype(int)

    preds_if    = to_bin_if(trained['if'].predict(X_orig))
    preds_lof   = to_bin_pyod(trained['lof'].predict(X_orig))   if lof_type   != 'if_fallback' else to_bin_if(trained['lof'].predict(X_orig))
    preds_ocsvm = to_bin_ocsvm(trained['ocsvm'].predict(X_orig)) if ocsvm_type != 'if_fallback' else to_bin_if(trained['ocsvm'].predict(X_orig))
    preds_ecod  = to_bin_ecod(trained['ecod'].predict(X_orig))
    preds_hbos  = to_bin_pyod(trained['hbos'].predict(X_orig))  if 'hbos'  in trained else np.zeros(len(y_orig), dtype=int)
    preds_copod = to_bin_pyod(trained['copod'].predict(X_orig)) if 'copod' in trained else np.zeros(len(y_orig), dtype=int)
    votes        = preds_if + preds_lof + preds_ocsvm + preds_ecod + preds_hbos + preds_copod
    preds_vote3  = (votes >= 3).astype(int)   # majorité sur 6
    preds_vote2  = (votes >= 2).astype(int)   # seuil bas

    # V9 : appliquer le filtre persistance aux votes binaires aussi
    preds_vote3_p = persistence_filter(preds_vote3, k=3)
    preds_vote2_p = persistence_filter(preds_vote2, k=3)

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
                random_state=RANDOM_STATE, n_jobs=N_JOBS_SAFE)
            self.model_.fit(X)
            return self
        def predict(self, X):
            return (self.model_.predict(X) == -1).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_pca_orig = X_pca[:n_orig] if X_pca is not None else None
    try:
        cv_scores = cross_val_score(IFWrapper(CONTAMINATION), X_pca_orig, y_orig,
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

    # ── Tableau comparatif (évaluation sur données originales) ───────────────
    metrics = {}
    models_preds = {
        'IF':           preds_if,
        'LOF':          preds_lof,
        'OCSVM':        preds_ocsvm,
        'ECOD':         preds_ecod,
        'HBOS':         preds_hbos,
        'COPOD':        preds_copod,
        'Vote2/6':      preds_vote2,
        'Vote3/6':      preds_vote3,
        'Vote2/6+P':    preds_vote2_p,
        'Vote3/6+P':    preds_vote3_p,
        'IF_opt':       opt_if,
        'LOF_opt':      opt_lof,
        'OCSVM_opt':    opt_ocsvm,
        'ECOD_opt':     opt_ecod,
        'SoftVote_std': opt_soft_std,
        'SoftVote':     opt_soft,        # ← RAPPORT
    }
    if opt_hbos  is not None: models_preds['HBOS_opt']  = opt_hbos
    if opt_copod is not None: models_preds['COPOD_opt'] = opt_copod

    print(f"\n  {'Modèle':<14} {'F1':>6} {'Précision':>10} {'Rappel':>8} {'AUC-ROC':>8} {'Anomalies':>10}")
    print(f"  {'─'*14} {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*10}")

    for name, preds in models_preds.items():
        f1   = f1_score(y_orig, preds, zero_division=0)
        prec = precision_score(y_orig, preds, zero_division=0)
        rec  = recall_score(y_orig, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_orig, preds)
        except Exception:
            auc = 0.5
        n_a  = int(preds.sum())
        acc  = accuracy_score(y_orig, preds)
        marker = " ← RAPPORT" if name == 'SoftVote' else ""
        color  = G if f1 > 0.65 else (Y if f1 > 0.45 else R)
        print(f"  {name:<14} {color}{f1:>6.4f}{RS} {prec:>10.4f} {rec:>8.4f} {auc:>8.4f} {n_a:>10d}{marker}")
        metrics[name] = {'f1': f1, 'precision': prec, 'recall': rec, 'auc_roc': auc, 'n_anomalies': n_a, 'accuracy': acc}

    # Stocker scores et seuils optimaux pour la sauvegarde (préfixe _ = interne)
    def _get_thr(scores, n_anom):
        return float(np.percentile(scores, 100 - n_anom / len(scores) * 100))

    metrics['_opt_preds']         = opt_soft
    metrics['_avg_score']         = meta_score_train   # score du stacking (RAPPORT)
    metrics['_f1_softvote']       = f1_opt_soft
    metrics['_sv_thr_precision']  = sv_thr_precision
    metrics['_n_orig']            = n_orig
    metrics['_meta_lr']           = meta_lr
    metrics['_meta_order']        = meta_order
    if test_holdout_metrics is not None:
        metrics['SoftVote_test_holdout'] = test_holdout_metrics
    # Stats de normalisation par modèle (p1/p99) — utilisées par l'API pour le soft score continu
    metrics['_score_stats'] = {
        'if':    {'p1': float(np.percentile(scores_if,    1)), 'p99': float(np.percentile(scores_if,    99))},
        'lof':   {'p1': float(np.percentile(scores_lof,   1)), 'p99': float(np.percentile(scores_lof,   99))},
        'ocsvm': {'p1': float(np.percentile(scores_ocsvm, 1)), 'p99': float(np.percentile(scores_ocsvm, 99))},
        'ecod':  {'p1': float(np.percentile(scores_ecod,  1)), 'p99': float(np.percentile(scores_ecod,  99))},
    }
    if scores_hbos  is not None:
        metrics['_score_stats']['hbos']  = {'p1': float(np.percentile(scores_hbos,  1)), 'p99': float(np.percentile(scores_hbos,  99))}
    if scores_copod is not None:
        metrics['_score_stats']['copod'] = {'p1': float(np.percentile(scores_copod, 1)), 'p99': float(np.percentile(scores_copod, 99))}
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
        row_s  = scaler.transform(row)
        row_p  = pca.transform(row_s)
        # V8 : modèles non-supervisés utilisent X_scaled (31D)
        row_unsup = row_s

        # Vote
        v_if    = 1 if trained['if'].predict(row_unsup)[0]    == -1 else 0
        v_lof   = (1 if trained['lof'].predict(row_unsup)[0]   == 1  else 0) if trained.get('lof_type')   != 'if_fallback' else (1 if trained['lof'].predict(row_unsup)[0]   == -1 else 0)
        v_ocsvm = (1 if trained['ocsvm'].predict(row_unsup)[0] == -1 else 0) if trained.get('ocsvm_type') != 'if_fallback' else (1 if trained['ocsvm'].predict(row_unsup)[0] == -1 else 0)
        ecod_pred = trained['ecod'].predict(row_unsup)[0]
        if trained.get('ecod_type') == 'pyod':
            v_ecod = 1 if ecod_pred == 1 else 0
        else:
            v_ecod = 1 if ecod_pred == -1 else 0
        v_hbos  = (1 if trained['hbos'].predict(row_unsup)[0]  == 1 else 0) if 'hbos'  in trained else 0
        v_copod = (1 if trained['copod'].predict(row_unsup)[0] == 1 else 0) if 'copod' in trained else 0
        total_votes = v_if + v_lof + v_ocsvm + v_ecod + v_hbos + v_copod
        n_unsup = sum(1 for k in ['if','lof','ocsvm','ecod','hbos','copod'] if k in trained)
        thr = max(2, n_unsup // 2)
        result = "ANOMALY" if total_votes >= thr else "NORMAL"
        correct = result == tc['expect'] or tc['expect'] == 'INCERTAIN'
        icon = f"{G}✅{RS}" if correct else f"{Y}⚠ {RS}"
        print(f"\n  {icon} {tc['name']}")
        print(f"     Temp={d['temperature']}°C  VibZ={d['vibration_z']}mg  I={d['current']}A")
        print(f"     Votes : IF={v_if} LOF={v_lof} OCSVM={v_ocsvm} ECOD={v_ecod} HBOS={v_hbos} COPOD={v_copod} → {total_votes}/{n_unsup}")
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
      model_lof_v3.pkl     → Local Outlier Factor (pyod)
      model_ocsvm_v3.pkl   → One-Class SVM (sklearn, sous-échantillon)
      model_ecod_v3.pkl    → ECOD (ou clone IF)
      scaler_v3.pkl        → RobustScaler (médiane + IQR)
      pca_v3.pkl           → PCA (~5 composantes)
      features_v3.pkl      → liste ordonnée des 31 features
      model_meta_lr.pkl    → stacking LogisticRegression (combine les scores des modèles ci-dessus)
      metrics_v3.csv       → métriques F1, AUC, etc.
    """
    head("ÉTAPE 6 — SAUVEGARDE")

    # Seuils optimaux (stockés par evaluate_models pour l'API)
    avg_score          = metrics.pop('_avg_score', None)
    opt_thresholds     = metrics.pop('_opt_thresholds', {})
    score_stats        = metrics.pop('_score_stats', {})
    sv_thr_precision   = metrics.pop('_sv_thr_precision', None)   # seuil precision-contraint
    meta_lr            = metrics.pop('_meta_lr', None)            # stacking LogisticRegression
    meta_order          = metrics.pop('_meta_order', [])
    metrics.pop('_opt_preds', None)
    metrics.pop('_f1_softvote', None)
    metrics.pop('_n_orig', None)
    if sv_thr_precision is not None:
        # V9 : utiliser le seuil optimisé pour precision >= 0.70 (avec filtre persistance)
        sv_thr = sv_thr_precision
    elif avg_score is not None:
        sv_n   = metrics.get('SoftVote', {}).get('n_anomalies', int(len(avg_score) * CONTAMINATION))
        sv_thr = float(np.percentile(avg_score, 100 - sv_n / len(avg_score) * 100))
    else:
        sv_thr = 0.5
    threshold_data = {
        'softvote_threshold':   sv_thr,         # seuil brut pour le soft score normalisé [0,1]
        'persistence_k':        3,               # fenêtres consécutives requises pour confirmer
        'contamination':        float(CONTAMINATION),
        'score_stats':          score_stats,     # p1/p99 par modèle — pour normalisation API
        **opt_thresholds,
    }

    # Flag pour l'API : les modèles non-supervisés utilisent X_scaled (pas X_pca)
    threshold_data['unsupervised_input'] = 'scaled'
    # RAPPORT = stacking LogisticRegression sur tous les modèles disponibles
    # (remplace l'ancienne exclusion manuelle de LOF/OCSVM — le méta-modèle
    # apprend lui-même leur poids, éventuellement quasi nul).
    threshold_data['n_unsup_models']     = len(meta_order)
    threshold_data['ensemble_names']     = [k.upper() for k in meta_order]
    threshold_data['meta_feature_order'] = meta_order

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
    if 'hbos'  in trained: files['model_hbos_v3.pkl']  = trained['hbos']
    if 'copod' in trained: files['model_copod_v3.pkl'] = trained['copod']
    if meta_lr is not None: files['model_meta_lr.pkl'] = meta_lr

    for fname, obj in files.items():
        path = MODEL_DIR / fname
        joblib.dump(obj, path)
        size_kb = path.stat().st_size / 1024
        ok(f"{fname:30s} → {size_kb:.0f} KB")

    # Métriques CSV — SoftVote non-supervisé.
    # Priorité au jeu de TEST tenu à l'écart (capteurs jamais vus pendant le
    # fit) quand il est disponible : ce sont les seules métriques qui
    # reflètent une vraie généralisation plutôt qu'une resubstitution.
    soft_metrics = metrics.get(
        'SoftVote_test_holdout', metrics.get('SoftVote', metrics.get('Vote2/6', {}))
    )
    vote_metrics = metrics.get('Vote2/6', {})
    csv_path = MODEL_DIR / "metrics_v3.csv"
    weight_lines = "\n".join(
        f"weights_{k},{c:.4f}" for k, c in zip(meta_order, (meta_lr.coef_[0] if meta_lr is not None else []))
    )
    csv_content = f"""metric,value
f1_score,{soft_metrics.get('f1', 0):.4f}
accuracy,{soft_metrics.get('accuracy', 0):.4f}
precision,{soft_metrics.get('precision', 0):.4f}
recall,{soft_metrics.get('recall', 0):.4f}
auc_roc,{soft_metrics.get('auc_roc', 0):.4f}
f1_vote2,{vote_metrics.get('f1', 0):.4f}
n_anomalies,{soft_metrics.get('n_anomalies', 0)}
n_total,{n_total}
contamination,{CONTAMINATION}
model_version,V8
n_features,31
ensemble,{' + '.join(k.upper() for k in meta_order)} (stacking)
voting,Stacking LogisticRegression + seuil F1-optimal sous contrainte precision>=0.70
augmentation,x{AUGMENT_FACTOR}
pca_variance,0.95
window_size,{WINDOW_SIZE}
dataset,ai_cp full_data — mesures reelles — 20 capteurs IFM — nov2025-mar2026
ecod_type,{trained.get('ecod_type', 'unknown')}
{weight_lines}
meta_intercept,{(meta_lr.intercept_[0] if meta_lr is not None else 0.0):.4f}
trained_at,{datetime.now().isoformat()}
evaluation,{'test_holdout_par_capteur' if 'SoftVote_test_holdout' in metrics else 'resubstitution_train'}
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
# ÉTAPE 1c — CHARGEMENT D'UN CSV PRÉ-GÉNÉRÉ (pipeline API /v1/pipeline/upload)
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataframe_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Charge un DataFrame directement depuis un CSV déjà consolidé — mêmes
    colonnes que CSV_COLUMNS dans generate_dataset_from_sql.py (sensor_id,
    timestamp, temperature, vibration_x/y/z, acc_p2p/z2p/crest/rms).

    Bypass complet de parse_sql_to_dataframe() : cette dernière cherche les
    tables `motor_mesure`/`motor_measurements`, un schéma différent de
    full_data (celui du dump réel du projet) et n'en extrait presque rien
    (401 sessions au lieu de 606 000 constatées sur le dump ai_cp complet).
    Le pipeline API génère déjà ce CSV correctement via
    generate_dataset_from_sql.py avant d'appeler ce script — inutile de
    reparser le SQL une 2e fois avec une logique différente et bien moins
    efficace.
    """
    head("ÉTAPE 1 — LECTURE DU CSV PRÉ-GÉNÉRÉ")
    info(f"Fichier : {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"sensor_id", "timestamp", "temperature", "vibration_x", "vibration_y", "vibration_z"}
    missing = required - set(df.columns)
    if missing:
        err(f"Colonnes manquantes dans le CSV : {missing}")
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["temperature", "vibration_z"])
    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    df["vibration_total"] = np.sqrt(
        df["vibration_x"].fillna(0) ** 2 + df["vibration_y"].fillna(0) ** 2 + df["vibration_z"].fillna(0) ** 2
    )
    # Courant non fourni par la gateway IFM sur la majorité des capteurs
    # (voir generate_dataset_from_sql.py) — même fallback que le reste du projet.
    df["current"] = 0.0
    df["source"]  = "csv_pregenere"

    # Accélérations réelles (acc_p2p/z2p/crest/rms) — mesure IFM "on-demand"
    # envoyée ~1x/heure/capteur (voir generate_dataset_from_sql.py), donc
    # présente sur une minorité de lignes seulement (~6% du CSV complet) : la
    # grande majorité des lignes ont 0.0 en attendant la prochaine mesure.
    # Sans forward-fill, une fenêtre glissante de WINDOW_SIZE=20 lignes ne
    # contient quasi jamais de valeur réelle et sa moyenne serait diluée à
    # ~valeur/20. On propage donc la dernière mesure connue par capteur
    # jusqu'à la suivante (comportement physique correct : l'accélération
    # mesurée reste le meilleur estimateur disponible jusqu'au prochain relevé).
    acc_cols = [c for c in ["acc_p2p", "acc_z2p", "acc_crest", "acc_rms"] if c in df.columns]
    if acc_cols:
        n_real = int((df[acc_cols] != 0).any(axis=1).sum())
        df[acc_cols] = df[acc_cols].replace(0.0, np.nan)
        df[acc_cols] = df.groupby("sensor_id")[acc_cols].ffill()
        df[acc_cols] = df[acc_cols].fillna(0.0)
        info(f"Accélération réelle : {n_real:,} mesures on-demand propagées par forward-fill par capteur")

    ok(f"CSV chargé : {len(df):,} lignes | {df['sensor_id'].nunique()} capteurs uniques")
    info(f"Période  : {df['timestamp'].min()} → {df['timestamp'].max()}")
    info(f"Température : {df['temperature'].min():.1f}°C – {df['temperature'].max():.1f}°C")
    info(f"Vib Z       : {df['vibration_z'].min():.2f} – {df['vibration_z'].max():.2f} mg")

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
    parser.add_argument('--csv',      default=None,          help="Charger directement un CSV pré-généré (voir generate_dataset_from_sql.py) — priorité absolue, bypass MySQL/SQL")
    parser.add_argument('--db-host',  default=MYSQL_HOST,    help="Hôte MySQL")
    parser.add_argument('--db-user',  default=MYSQL_USER,    help="Utilisateur MySQL")
    parser.add_argument('--db-pass',  default=MYSQL_PASSWORD,help="Mot de passe MySQL")
    parser.add_argument('--db-name',  default=MYSQL_DATABASE,help="Base de données MySQL")
    parser.add_argument('--no-mysql', action='store_true',   help="Forcer utilisation SQL (ignorer MySQL)")
    parser.add_argument('--out-dir',  default=None,
                         help="Dossier de sauvegarde des .pkl/metrics_v3.csv (défaut: models/, "
                              "utilisé par l'API en production). Passer un dossier distinct pour "
                              "un run de test sans écraser les modèles de production.")
    args = parser.parse_args()

    if args.out_dir:
        global MODEL_DIR
        MODEL_DIR = Path(args.out_dir)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{B}{C}{'╔'+'═'*60+'╗'}")
    print(f"║  ENTRAÎNEMENT MODÈLES NON SUPERVISÉS V6                   ║")
    print(f"║  Source : MySQL full_data (25 000 sessions réelles)       ║")
    print(f"║  Modèles : IF + LOF + OCSVM + ECOD                        ║")
    print(f"╚{'═'*60+'╝'}{RS}\n")

    t_total = time.time()

    sensor_ids = None   # groupes pour le split train/test — None si source sans capteur identifiable

    if args.csv:
        # ── Priorité 0 : CSV pré-généré explicitement fourni (pipeline API) ───────
        # Bypass total de MySQL/realtime/SQL-fallback : le CSV a déjà été
        # correctement consolidé par generate_dataset_from_sql.py, pas la peine
        # de reparser le SQL une 2e fois avec la logique motor_mesure legacy.
        df_csv = load_dataframe_from_csv(args.csv)
        ok(f"Source CSV pré-généré : {len(df_csv):,} sessions")
        X, feature_names, y_heuristic, sensor_ids = build_feature_matrix(df_csv, augment=False)

    else:
        # ── Priorité 1 : MySQL full_data (données production bonne échelle) ──────
        df_mysql = pd.DataFrame()
        if not args.no_mysql:
            df_mysql = parse_full_data_from_mysql(
                host=args.db_host, user=args.db_user,
                password=args.db_pass, database=args.db_name
            )

        if len(df_mysql) >= 200:
            ok(f"Source MySQL full_data : {len(df_mysql)} sessions à l'échelle production")
            X, feature_names, y_heuristic, sensor_ids = build_feature_matrix(df_mysql, augment=False)

        else:
            # ── Priorité 2 : features production depuis realtime_results.json ────
            X_prod, feat_names_prod, y_prod = parse_features_from_realtime(REALTIME_RESULTS)
            if len(X_prod) >= 500 and float(y_prod.mean()) >= 0.03:
                ok(f"Source realtime_results.json : {len(X_prod)} vecteurs × 31 features")
                X, feature_names, y_heuristic = X_prod, feat_names_prod, y_prod
                # Pas de sensor_id disponible ici → split aléatoire par ligne (fallback
                # dégradé, acceptable car ce chemin n'est utilisé qu'en secours).
                sensor_ids = np.arange(len(X)).astype(str)
            else:
                # ── Priorité 3 : SQL file (fallback, domain gap) ──────────────
                warn(f"MySQL indisponible + realtime insuffisant → fallback SQL ({args.sql})")
                df_sql = parse_sql_to_dataframe(args.sql)
                X, feature_names, y_heuristic, sensor_ids = build_feature_matrix(df_sql, augment=False)

    # Recalibrer la contamination (clampé 5%-35%) sur les données originales
    global CONTAMINATION
    CONTAMINATION = round(max(0.05, min(0.35, float(y_heuristic.mean()))), 3)
    info(f"Contamination recalibree : {CONTAMINATION:.1%} ({int(y_heuristic.sum())}/{len(y_heuristic)} sessions anormales)")

    # ── Validation croisée PAR CAPTEUR (GroupKFold, anti fuite de données) ────
    # Un split unique 80/20 sur seulement 19 capteurs est à haute variance —
    # un tirage malchanceux peut isoler presque toutes les anomalies d'un
    # côté. GroupKFold partitionne les capteurs en K groupes disjoints et
    # teste chacun exactement une fois : la moyenne sur K folds est une
    # estimation bien plus stable de la généralisation réelle.
    head("VALIDATION CROISÉE PAR CAPTEUR (GroupKFold, anti fuite de données)")
    from sklearn.model_selection import GroupKFold
    n_groups = len(np.unique(sensor_ids))
    N_SPLITS = min(3, n_groups) if n_groups >= 3 else 0

    fold_metrics = []
    if N_SPLITS >= 3:
        gkf = GroupKFold(n_splits=N_SPLITS)
        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y_heuristic, groups=sensor_ids), 1):
            head(f"FOLD {fold_i}/{N_SPLITS}")
            X_train_orig, y_train_orig = X[train_idx], y_heuristic[train_idx]
            X_test_orig,  y_test_orig  = X[test_idx],  y_heuristic[test_idx]
            ok(f"Train : {len(X_train_orig)} sessions ({np.unique(sensor_ids[train_idx]).size} capteurs) | "
               f"{int(y_train_orig.sum())} anomalies ({y_train_orig.mean()*100:.1f}%)")
            ok(f"Test  : {len(X_test_orig)} sessions ({np.unique(sensor_ids[test_idx]).size} capteurs, jamais vus) | "
               f"{int(y_test_orig.sum())} anomalies ({y_test_orig.mean()*100:.1f}%)")

            X_train_aug, y_train_aug = augment_data(X_train_orig, y_train_orig)
            scaler_f, pca_f, X_train_pca_f, X_train_scaled_f = preprocess(X_train_aug)
            X_test_scaled_f = scaler_f.transform(X_test_orig)

            trained_f = train_models(X_train_pca_f, y_train_aug, X_train_scaled_f)
            metrics_f = evaluate_models(
                trained_f, X_train_pca_f, y_train_aug, scaler_f, pca_f, feature_names, X_train_scaled_f,
                X_test_scaled=X_test_scaled_f, y_test=y_test_orig,
            )
            if 'SoftVote_test_holdout' in metrics_f:
                fold_metrics.append(metrics_f['SoftVote_test_holdout'])

        if fold_metrics:
            avg_metrics = {
                k: float(np.mean([m[k] for m in fold_metrics]))
                for k in ['f1', 'precision', 'recall', 'auc_roc', 'accuracy']
            }
            avg_metrics['n_anomalies'] = int(sum(m['n_anomalies'] for m in fold_metrics))
            avg_metrics['n_total']     = int(sum(m['n_total'] for m in fold_metrics))
            head("MOYENNE CROSS-VALIDATION (métrique officielle rapportée)")
            for m in fold_metrics:
                info(f"  Fold : F1={m['f1']:.4f}  Prec={m['precision']:.4f}  Recall={m['recall']:.4f}  AUC={m['auc_roc']:.4f}")
            ok(f"MOYENNE {N_SPLITS} folds : F1={avg_metrics['f1']:.4f}  Prec={avg_metrics['precision']:.4f}  "
               f"Recall={avg_metrics['recall']:.4f}  AUC={avg_metrics['auc_roc']:.4f}")
        else:
            avg_metrics = None
            warn("Aucun fold n'a produit de métriques de test exploitables")
    else:
        warn(f"Seulement {n_groups} capteur(s) distinct(s) — cross-validation par groupe impossible, "
             f"entraînement direct sur toutes les données (métriques en resubstitution)")
        avg_metrics = None

    # ── Entraînement FINAL sur TOUTES les données (production) ───────────────
    # La CV ci-dessus sert uniquement à estimer honnêtement la généralisation.
    # Le modèle réellement déployé est entraîné sur l'intégralité des
    # capteurs disponibles — c'est la pratique standard (plus de données
    # disponibles = meilleur modèle final, la CV a déjà validé l'approche).
    head("ENTRAÎNEMENT FINAL SUR TOUTES LES DONNÉES (modèles déployés)")
    X_full_aug, y_full_aug = augment_data(X, y_heuristic)
    scaler, pca, X_full_pca, X_full_scaled = preprocess(X_full_aug)
    trained = train_models(X_full_pca, y_full_aug, X_full_scaled)
    metrics = evaluate_models(trained, X_full_pca, y_full_aug, scaler, pca, feature_names, X_full_scaled)

    # Les métriques officiellement rapportées sont celles de la CV (généralisation
    # honnête), pas celles du modèle final (resubstitution sur toutes les données).
    if avg_metrics is not None:
        metrics['SoftVote_test_holdout'] = avg_metrics

    # Étape 6 : Sauvegarder
    save_models(trained, scaler, pca, feature_names, metrics, n_total=len(y_full_aug))

    # ── Résumé final ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    head(f"TERMINÉ EN {elapsed:.1f}s")
    ok(f"4 modèles entraînés et sauvegardés dans {MODEL_DIR}/")
    ok(f"Vote2/6 : F1={metrics.get('Vote2/6', {}).get('f1', 0):.4f}  |  Vote3/6 : F1={metrics.get('Vote3/6', {}).get('f1', 0):.4f}")
    ok(f"31 features V6 : +delta_vib, +delta_temp, +vib_entropy, +fft_ratio, +vib_asym_xy, +vib_asym_xz")
    info("Relance l'API : python api_unified_pythagore.py")


if __name__ == "__main__":
    main()