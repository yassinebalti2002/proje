"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  API Unifiée — Maintenance Prédictive Roulements                            ║
║  PFE — Surveillance de 20 capteurs IFM — Novation City                      ║
║                                                                              ║
║  Port : 8000                                                                 ║
║  Docs : http://localhost:8000/docs                                           ║
║                                                                              ║
║  Endpoints principaux :                                                      ║
║    POST /v1/predict              → Détection anomalie (IF+LOF+OCSVM+ECOD)   ║
║    POST /v1/predict-rul          → Estimation RUL (Remaining Useful Life)   ║
║    GET  /v1/health-score/{id}    → Score santé moteur                        ║
║    GET  /v1/history/{id}         → Historique prédictions par capteur  [NEW] ║
║    GET  /v1/alert-level/{id}     → Niveau d'alerte actuel capteur      [NEW] ║
║    GET  /health                  → Health check                              ║
║    GET  /metrics                 → Métriques modèle V3 (F1=0.401, AUC=0.683)║
║    GET  /sensors                 → Liste capteurs depuis full_data           ║
║    GET  /anomalies               → Anomalies filtrées                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
from collections import deque
from scipy.stats import entropy as sp_entropy

# ── Module d'alertes externes (email / webhook / SMS) ─────────────────────────
try:
    from alert_manager import AlertManager
    _alert_manager = AlertManager()
    ALERTS_ENABLED = True
except Exception as _e:
    _alert_manager = None
    ALERTS_ENABLED = False
    logging.getLogger(__name__).warning(
        f"AlertManager non disponible : {_e} — les alertes externes sont désactivées"
    )

# ── Pipeline traitement du signal (FFT, analyse spectrale, défauts roulements) ─
try:
    from signal_processing import extract_spectral_features, BearingFaultDetector, full_signal_pipeline
    SIGNAL_PROCESSING_OK = True
except Exception as _sp_e:
    SIGNAL_PROCESSING_OK = False
    logging.getLogger(__name__).warning(
        f"signal_processing non disponible : {_sp_e}"
    )

# ── Modèle RUL ML dédié (GradientBoosting entraîné sur courbes de dégradation) ─
try:
    from train_rul_model import RULPredictor
    _rul_predictor = RULPredictor()
    RUL_ML_ENABLED = _rul_predictor.load()
except Exception as _rul_e:
    _rul_predictor = None
    RUL_ML_ENABLED = False
    logging.getLogger(__name__).warning(
        f"RULPredictor ML non disponible : {_rul_e} — utilisation heuristique"
    )

# ── Module de reporting ────────────────────────────────────────────────────────
try:
    from reporting_module import generate_html_report, generate_json_report, save_report
    REPORTING_OK = True
except Exception as _rep_e:
    REPORTING_OK = False
    logging.getLogger(__name__).warning(f"reporting_module non disponible : {_rep_e}")

warnings.filterwarnings("ignore")

# ── Version unique — utilisée partout ─────────────────────────────────────────
API_VERSION = "3.1.0"

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from contextlib import asynccontextmanager
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    print("FastAPI non installé. Lance : pip install fastapi uvicorn pydantic")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [API] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  CHEMINS
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_DIR  = Path(__file__).parent
MODEL_DIR    = PROJECT_DIR / "models"
DATA_DIR     = PROJECT_DIR / "data"

# Modèles V3 (ensemble non supervisé)
MODEL_IF        = MODEL_DIR / "model_if_v3.pkl"
MODEL_LOF       = MODEL_DIR / "model_lof_v3.pkl"
MODEL_OCSVM     = MODEL_DIR / "model_ocsvm_v3.pkl"
MODEL_ECOD      = MODEL_DIR / "model_ecod_v3.pkl"
SCALER_PATH     = MODEL_DIR / "scaler_v3.pkl"
PCA_PATH        = MODEL_DIR / "pca_v3.pkl"
FEATURES_PATH   = MODEL_DIR / "features_v3.pkl"
THRESHOLD_PATH  = MODEL_DIR / "threshold_v3.pkl"
METRICS_PATH    = MODEL_DIR / "metrics_v3.csv"
# Fallback si metrics_v2 absent → chercher metrics_v3
if not METRICS_PATH.exists():
    METRICS_PATH = MODEL_DIR / "metrics_v3.csv"
RESULTS_PATH = DATA_DIR  / "results_v2.csv"

# ══════════════════════════════════════════════════════════════════════════════
#  ÉTAT GLOBAL (modèles + historique anomalies par moteur)
# ══════════════════════════════════════════════════════════════════════════════

models        = {}    # {"if": ..., "lof": ..., "ocsvm": ..., "ecod": ...}
scaler        = None
pca           = None
features_list = []
df_results    = None
thresholds    = {}    # seuils optimaux du SoftVote chargés depuis threshold_v3.pkl

# Historique glissant des scores d'anomalie par sensor_id (pour RUL)
# Format : {sensor_id: deque([(timestamp, anomaly_score, confidence), ...])}
anomaly_history: dict = {}
HISTORY_WINDOW = 50   # Nombre de mesures gardées en mémoire par moteur

# Baseline par capteur — pour normaliser le health score relatif (semaine 2)
# Calculé sur les 20 premières mesures de chaque capteur
sensor_baseline: dict = {}
BASELINE_SAMPLES = 20

# Persistence de l'historique sur disque (survit au redémarrage)
HISTORY_PATH = Path("anomaly_history_persist.json")
PERSIST_INTERVAL = 300   # sauvegarder toutes les 5 minutes
_last_persist: float = 0.0

# Fenêtres glissantes serveur pour /v1/iot-predict (IoT sans accès base de données)
# {sensor_id: deque([MeasurePoint-like dict, ...], maxlen=IOT_WINDOW_SIZE)}
iot_windows: dict = {}
IOT_WINDOW_SIZE = 10


def save_history_to_disk():
    """Sauvegarde anomaly_history sur disque pour survivre au redémarrage."""
    global _last_persist
    import time as _time
    now = _time.time()
    if now - _last_persist < PERSIST_INTERVAL:
        return
    try:
        data = {
            sid: [dict(e) for e in list(dq)]
            for sid, dq in anomaly_history.items()
        }
        HISTORY_PATH.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        _last_persist = now
        log.debug(f"Historique persisté → {HISTORY_PATH} ({len(data)} capteurs)")
    except Exception as e:
        log.warning(f"Persistence historique échouée : {e}")


IFM_KNOWN_IDS = {
    "07da47b8","0ff416d2","2c6254af","3a782f1b","4b5e4b32",
    "53cb61b2","68c11f06","6e0c1740","718fd2af","8f7f2f7e",
    "91d92804","99695e98","a6a46be1","aa7b02a1","b2acdf45",
    "bc59bf5f","d9508e77","eb084747","f48c25f9",
}

def load_history_from_disk():
    """Restaure anomaly_history depuis le fichier de persistence au démarrage.
    Filtre uniquement les 20 capteurs IFM connus pour éviter la pollution
    par d'anciens runs (port*, simulateur, tests)."""
    global anomaly_history, sensor_baseline
    if not HISTORY_PATH.exists():
        return
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        loaded = 0
        for sid, entries in data.items():
            if sid not in IFM_KNOWN_IDS:
                continue  # ignorer port*, simulateur, etc.
            anomaly_history[sid] = deque(entries, maxlen=HISTORY_WINDOW)
            scores = [e["score"] for e in entries[:BASELINE_SAMPLES]]
            if len(scores) >= 5:
                sensor_baseline[sid] = float(np.mean(scores))
            loaded += 1
        log.info(f"Historique restauré depuis {HISTORY_PATH} ({loaded}/{len(data)} capteurs IFM valides)")
    except Exception as e:
        log.warning(f"Restauration historique échouée : {e}")


def load_all_models():
    global models, scaler, pca, features_list, df_results, thresholds

    log.info("Chargement des modèles V3...")

    missing = []
    for name, path in [
        ("if", MODEL_IF), ("lof", MODEL_LOF),
        ("ocsvm", MODEL_OCSVM), ("ecod", MODEL_ECOD),
        ("scaler", SCALER_PATH), ("pca", PCA_PATH), ("features", FEATURES_PATH)
    ]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        log.warning(f"Fichiers manquants : {missing}")
        log.warning("Lance d'abord : python train_model_v3_unsupervised.py")
    else:
        models["if"]    = joblib.load(MODEL_IF)
        models["lof"]   = joblib.load(MODEL_LOF)
        models["ocsvm"] = joblib.load(MODEL_OCSVM)
        models["ecod"]  = joblib.load(MODEL_ECOD)
        scaler          = joblib.load(SCALER_PATH)
        pca             = joblib.load(PCA_PATH)
        features_list   = joblib.load(FEATURES_PATH)
        if THRESHOLD_PATH.exists():
            thresholds = joblib.load(THRESHOLD_PATH)
            log.info(f"✅ Seuils optimaux chargés : softvote_thr={thresholds.get('softvote_threshold', 0.5):.4f}")
        log.info(f"✅ 4 modèles chargés | Features: {len(features_list)} | PCA: {pca.n_components_}")

    if RESULTS_PATH.exists():
        df_results = pd.read_csv(RESULTS_PATH)
        log.info(f"✅ Résultats historiques : {len(df_results)} lignes")


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS PYDANTIC
# ══════════════════════════════════════════════════════════════════════════════

class MeasurePoint(BaseModel):
    """Une mesure capteur à un instant t (compatible full_data JSON)."""
    timestamp:   Optional[str]   = None
    temperature: Optional[float] = Field(None, ge=-20.0, le=150.0,
                    description="Température en °C (plage physique : -20 à 150°C)")
    vibration_x: Optional[float] = Field(None, ge=0.0, le=5000.0,
                    description="Vibration RMS axe X en mg (0 à 5000 mg)")
    vibration_y: Optional[float] = Field(None, ge=0.0, le=5000.0,
                    description="Vibration RMS axe Y en mg (0 à 5000 mg)")
    vibration_z: Optional[float] = Field(None, ge=0.0, le=5000.0,
                    description="Vibration RMS axe Z en mg (0 à 5000 mg)")
    current:     Optional[float] = Field(None, ge=0.0, le=500.0,
                    description="Courant moteur en A (0 à 500 A)")
    power:       Optional[float] = Field(None, ge=0.0, le=100000.0,
                    description="Puissance en W")
    vitesse:     Optional[float] = Field(None, ge=0.0, le=10000.0,
                    description="Vitesse en RPM (0 à 10 000 tr/min)")
    a_rms:       Optional[float] = Field(None, ge=0.0, le=5000.0)
    crest:       Optional[float] = Field(None, ge=0.0, le=50.0)
    # ── Nouvelles features accélération IFM (gph='acceleration') ──────────
    acc_p2p:     Optional[float] = Field(None, ge=0.0, le=30000.0,
                    description="Accélération Peak-to-Peak axe Y (mg)")
    acc_z2p:     Optional[float] = Field(None, ge=0.0, le=15000.0,
                    description="Accélération Zero-to-Peak axe Y (mg)")
    acc_crest:   Optional[float] = Field(None, ge=0.0, le=5000.0,
                    description="Facteur de crête accélération axe Y")
    acc_rms:     Optional[float] = Field(None, ge=0.0, le=5000.0,
                    description="Accélération RMS axe Y (mg)")


class PredictRequest(BaseModel):
    """Corps POST /v1/predict — compatible Node-RED + full_data."""
    sensor_id: str = Field(..., example="8f7f2f7e")
    motor_id:  Optional[str] = Field(None, example="Motor_1604")
    history:   List[MeasurePoint] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    sensor_id:      str
    motor_id:       Optional[str]
    timestamp:      str
    prediction:     str           # "ANOMALY" ou "NORMAL"
    is_anomaly:     bool
    confidence:     float         # 0.0 – 1.0
    votes:          int           # 0 – 4
    risk_level:     str           # "FAIBLE" | "MODÉRÉ" | "CRITIQUE"
    anomaly_score:  float
    individual_models: dict
    features: dict


class RULRequest(BaseModel):
    """Corps POST /v1/predict-rul."""
    sensor_id:     str            = Field(..., example="8f7f2f7e")
    motor_id:      Optional[str]   = Field(None, example="Motor_1604")
    prediction:    Optional[str]   = Field("NORMAL", example="NORMAL")
    votes:         Optional[int]   = Field(0, example=0)
    confidence:    Optional[float] = Field(0.0, ge=0.0, le=1.0)
    risk_level:    Optional[str]   = Field("OK", example="OK")
    anomaly_score: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    history:       List[MeasurePoint] = Field(..., min_length=3,
                       description="Minimum 3 mesures pour estimer la tendance")


class RULResponse(BaseModel):
    sensor_id:       str
    motor_id:        Optional[str]
    timestamp:       str
    rul_hours:       float        # Heures estimées avant défaillance
    rul_days:        float        # Jours estimés
    degradation_rate: float       # % de dégradation par mesure
    health_score:    float        # Score santé 0–100
    confidence:      str          # "HAUTE" | "MOYENNE" | "FAIBLE"
    alert_level:     str          # "OK" | "ATTENTION" | "URGENT" | "CRITIQUE"
    recommendation:  str
    trend: dict                   # Détail des tendances par feature


class IoTMeasurementRequest(BaseModel):
    """Mesure IoT brute — format direct capteur IFM / gateway, sans base de données.

    Le collègue envoie une mesure par session (temperature + vibration X/Y/Z).
    L'historique glissant est géré côté serveur (fenêtre de 10 mesures par capteur).
    La prédiction ET le RUL sont retournés en un seul appel.
    """
    sensor_id:   str   = Field(..., example="8f7f2f7e",
                    description="ID capteur IFM (hex 8 chars, ex: 8f7f2f7e)")
    motor_id:    Optional[str]  = Field(None, example="Motor_8f7f2f7e")
    timestamp:   Optional[str]  = Field(None,
                    description="Horodatage ISO 8601 — généré automatiquement si absent")
    temperature: float = Field(..., ge=-20.0, le=150.0,
                    description="Température en °C (issue de gph='temperature')")
    vibration_x: float = Field(..., ge=0.0,   le=5000.0,
                    description="Vibration RMS axe X en mg (issue de gph='vibration_x')")
    vibration_y: float = Field(..., ge=0.0,   le=5000.0,
                    description="Vibration RMS axe Y en mg (issue de gph='vibration_y')")
    vibration_z: float = Field(..., ge=0.0,   le=5000.0,
                    description="Vibration RMS axe Z en mg (issue de gph='temperature' → Vibration.RMS.Z)")
    current:     Optional[float] = Field(0.0, ge=0.0, le=500.0,
                    description="Courant moteur en A (0 si non disponible)")
    acc_p2p:     Optional[float] = Field(None, ge=0.0, le=30000.0,
                    description="Accélération Peak-to-Peak (mg) — optionnel")
    acc_rms:     Optional[float] = Field(None, ge=0.0, le=5000.0,
                    description="Accélération RMS (mg) — optionnel")
    acc_crest:   Optional[float] = Field(None, ge=0.0, le=5000.0,
                    description="Facteur de crête accélération — optionnel")
    acc_z2p:     Optional[float] = Field(None, ge=0.0, le=15000.0,
                    description="Accélération Zero-to-Peak (mg) — optionnel")


class IoTPredictResponse(BaseModel):
    """Réponse /v1/iot-predict : prédiction anomalie + RUL en un seul appel."""
    sensor_id:      str
    motor_id:       Optional[str]
    timestamp:      str
    window_size:    int           # Nombre de mesures en mémoire côté serveur
    # ── Prédiction anomalie ──────────────────────────────────────────────────
    prediction:     str           # "ANOMALY" | "NORMAL"
    is_anomaly:     bool
    confidence:     float
    votes:          int
    risk_level:     str           # "FAIBLE" | "MODÉRÉ" | "ÉLEVÉ" | "CRITIQUE"
    anomaly_score:  float
    individual_models: dict
    # ── RUL (None si moins de 3 mesures accumulées) ──────────────────────────
    rul_hours:      Optional[float]
    rul_days:       Optional[float]
    health_score:   Optional[float]
    alert_level:    Optional[str]
    recommendation: Optional[str]
    # ── Features extraites ───────────────────────────────────────────────────
    features:       dict


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS COMMUNS
# ══════════════════════════════════════════════════════════════════════════════

def safe_mean(lst):
    return float(np.mean(lst)) if lst else np.nan

def safe_std(lst):
    return float(np.std(lst)) if len(lst) > 1 else 0.0

def safe_rms(lst):
    arr = np.array(lst)
    return float(np.sqrt(np.mean(arr**2))) if len(arr) > 0 else np.nan

def safe_kurtosis(lst):
    from scipy.stats import kurtosis as sp_kurt
    return float(sp_kurt(lst)) if len(lst) >= 4 else 0.0

def safe_crest(lst):
    arr = np.array(lst)
    rms = np.sqrt(np.mean(arr**2))
    return float(np.max(np.abs(arr)) / (rms + 1e-9)) if len(arr) > 0 else np.nan

def safe_trend(lst):
    """Pente linéaire (régression deg 1). Positive = dégradation croissante."""
    if len(lst) >= 3:
        return float(np.polyfit(range(len(lst)), lst, 1)[0])
    return 0.0

def norm01(x, lo, hi):
    if np.isnan(x): return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo + 1e-9)))

def vib_total_pythagorean(rms_x: float, rms_y: float, rms_z: float) -> float:
    """
    Norme vibratoire totale 3D — Théorème de Pythagore généralisé.

    Formule :  V_total = √(X² + Y² + Z²)

    Base industrielle : ISO 10816-3 & ISO 20816
    Avantage : capture l'énergie vibratoire globale indépendamment
               de l'orientation du défaut de roulement.

    Seuils typiques roulements industriels (mg) :
        < 400  mg  → NORMAL
        400–800 mg → ATTENTION
        800–1200 mg → URGENT
        > 1200 mg  → CRITIQUE
    """
    x = rms_x if not np.isnan(rms_x) else 0.0
    y = rms_y if not np.isnan(rms_y) else 0.0
    z = rms_z if not np.isnan(rms_z) else 0.0
    return float(np.sqrt(x**2 + y**2 + z**2))


def medfilt_list(lst: list, k: int = 3) -> list:
    """Filtre median sur une liste — supprime les pics ponctuels (spikes)."""
    if len(lst) < k:
        return lst
    arr = np.array(lst, dtype=float)
    half = k // 2
    result = arr.copy()
    for i in range(half, len(arr) - half):
        result[i] = np.median(arr[i - half: i + half + 1])
    return result.tolist()


def signal_entropy_api(lst: list) -> float:
    """Entropie de Shannon — irregularite du signal."""
    if len(lst) < 4:
        return 0.0
    arr = np.array(lst, dtype=float)
    counts, _ = np.histogram(arr, bins=min(10, len(arr)))
    return float(sp_entropy(counts + 1e-9))


def fft_ratio_api(lst: list) -> float:
    """Ratio energie FFT dominante / totale — periodicite anormale."""
    if len(lst) < 4:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(np.array(lst, dtype=float)))
    total = np.sum(fft_vals**2) + 1e-9
    return float(np.max(fft_vals**2) / total)


def extract_features(history: List[MeasurePoint]) -> dict:
    """
    Extrait les 31 features depuis l'historique de mesures (V6).
    25 features de base + 6 nouvelles (FFT, delta, entropie, asymetrie).

    V6 — Ameliorations :
    - Filtre median anti-spike avant extraction (k=3)
    - 6 nouvelles features : delta_vib, delta_temp, vib_entropy,
      fft_ratio, vib_asym_xy, vib_asym_xz
    - Fallback vib_x/vib_y sur vib_z si axes absents
    """
    # Filtre median anti-spike (supprime les pics ponctuels non representatifs)
    temps_raw = [h.temperature for h in history if h.temperature is not None]
    vib_z_raw = [h.vibration_z for h in history if h.vibration_z is not None]
    temps = medfilt_list(temps_raw)
    vib_z = medfilt_list(vib_z_raw)

    vib_x_raw = [h.vibration_x for h in history
                 if h.vibration_x is not None and h.vibration_x > 0]
    vib_x = medfilt_list(vib_x_raw) if vib_x_raw else vib_z

    vib_y_raw = [h.vibration_y for h in history
                 if h.vibration_y is not None and h.vibration_y > 0]
    vib_y = medfilt_list(vib_y_raw) if vib_y_raw else vib_z

    acc_p2p_l  = [h.acc_p2p   for h in history if getattr(h,'acc_p2p',None)   is not None]
    acc_z2p_l  = [h.acc_z2p   for h in history if getattr(h,'acc_z2p',None)   is not None]
    acc_crest_l= [h.acc_crest for h in history if getattr(h,'acc_crest',None) is not None]
    acc_rms_l  = [h.acc_rms   for h in history if getattr(h,'acc_rms',None)   is not None]

    # ── Diagnostic qualite donnees (features structurellement nulles) ──────
    _log = logging.getLogger(__name__)
    if not acc_p2p_l and not acc_z2p_l and not acc_crest_l and not acc_rms_l:
        _log.debug(
            'LIMIT | acc_p2p=0, acc_z2p=0, acc_crest=0, acc_rms=0 '
            '-- Gateway IFM AL1352 ne transmet pas ces accelerations dans le '
            'flux de consolidation multi-lignes. Ces 4 features sont toujours '
            '0.0 : variance nulle, absorbees par PCA sans effet discriminant.'
        )
    current_vals = [h.current for h in history if h.current is not None]
    if not current_vals:
        _log.debug(
            'LIMIT | current_mean=0.0 -- Capteurs IFM VVB001/VSE002 : '
            'pas de mesure courant electrique. Aucun capteur de courant installe. '
            'La feature current_mean neutralise 15%% du health_score.'
        )

    feat = {
        # Thermique (4)
        "temp_mean":    safe_mean(temps),
        "temp_std":     safe_std(temps),
        "temp_trend":   safe_trend(temps),
        "temp_cur":     temps[-1] if temps else np.nan,
        # Vibration Z (6)
        "vib_z_mean":   safe_mean(vib_z),
        "vib_z_std":    safe_std(vib_z),
        "vib_z_rms_w":  safe_rms(vib_z),
        "vib_z_kurt":   safe_kurtosis(vib_z),
        "vib_z_crest":  safe_crest(vib_z),
        "vib_z_cur":    vib_z[-1] if vib_z else np.nan,
        # Vibration X (4)
        "vib_x_mean":   safe_mean(vib_x),
        "vib_x_std":    safe_std(vib_x),
        "vib_x_rms_w":  safe_rms(vib_x),
        "vib_x_kurt":   safe_kurtosis(vib_x),
        # Vibration Y (4)
        "vib_y_mean":   safe_mean(vib_y),
        "vib_y_std":    safe_std(vib_y),
        "vib_y_rms_w":  safe_rms(vib_y),
        "vib_y_kurt":   safe_kurtosis(vib_y),
        # Ratios inter-axes (conserves pour compatibilite)
        "vib_xy_ratio": safe_mean(vib_x) / (safe_mean(vib_y) + 1e-9)
                        if vib_x and vib_y else np.nan,
        "vib_xz_ratio": safe_mean(vib_x) / (safe_mean(vib_z) + 1e-9)
                        if vib_x and vib_z else np.nan,
        # Courant moteur
        "current_mean": safe_mean([h.current for h in history if h.current is not None]) or 0.0,
        "current_std":  safe_std([h.current  for h in history if h.current is not None]) or 0.0,
    }

    # Vibration totale 3D — Théorème de Pythagore
    rms_x = feat.get("vib_x_rms_w", 0.0) or 0.0
    rms_y = feat.get("vib_y_rms_w", 0.0) or 0.0
    rms_z = feat.get("vib_z_rms_w", 0.0) or 0.0
    feat["vib_total"] = round(vib_total_pythagorean(rms_x, rms_y, rms_z), 4)

    VIB_TOTAL_MAX = float(np.sqrt(3) * 1500)
    temp_n  = norm01(feat["temp_mean"]  if feat["temp_mean"]  else 35, 25, 65)
    vib_n   = norm01(feat["vib_total"]  if feat["vib_total"]  else 0,  0, VIB_TOTAL_MAX)
    kurt_n  = norm01(feat["vib_z_kurt"] if feat["vib_z_kurt"] else 0,  0, 10)
    # Note : capteurs IFM VVB001 ne transmettent pas le courant → cur_n=0 systématiquement
    # Redistribution des poids : 0.35 Temp + 0.35 Vib + 0.30 Kurtosis = 1.00
    feat["health_score"] = round(
        100 * (1 - 0.35*temp_n - 0.35*vib_n - 0.30*kurt_n), 1
    )

    # Accélération IFM (4)
    feat["acc_p2p"]   = safe_mean(acc_p2p_l)   if acc_p2p_l   else 0.0
    feat["acc_z2p"]   = safe_mean(acc_z2p_l)   if acc_z2p_l   else 0.0
    feat["acc_crest"] = safe_mean(acc_crest_l)  if acc_crest_l else 0.0
    feat["acc_rms"]   = safe_mean(acc_rms_l)    if acc_rms_l   else 0.0

    # ── V6 : 6 nouvelles features ─────────────────────────────────────────
    mid = max(1, len(vib_z) // 2)
    feat["delta_vib"]   = float(np.mean(vib_z[mid:]) - np.mean(vib_z[:mid])) if len(vib_z) >= 4 else 0.0
    feat["delta_temp"]  = float(np.mean(temps[mid:]) - np.mean(temps[:mid])) if len(temps) >= 4 else 0.0
    feat["vib_entropy"] = signal_entropy_api(vib_z)
    feat["fft_ratio"]   = fft_ratio_api(vib_z)
    vx_m = safe_mean(vib_x) or 0.0
    vy_m = safe_mean(vib_y) or 0.0
    vz_m = safe_mean(vib_z) or 0.0
    feat["vib_asym_xy"] = float(abs(vx_m - vy_m) / (vx_m + vy_m + 1e-9))
    feat["vib_asym_xz"] = float(abs(vx_m - vz_m) / (vx_m + vz_m + 1e-9))

    return feat


def run_ensemble(X_raw: np.ndarray) -> dict:
    """
    Applique scaler -> PCA -> SoftVote (IF+OCSVM+ECOD, seuil optimal).
    Fallback sur Vote2/4 binaire si threshold_v3.pkl absent.
    """
    if not models:
        return {"votes": 0, "label": "NORMAL", "confidence": 0.0,
                "is_anomaly": False,
                "individual": {"IF": "N/A", "LOF": "N/A",
                               "OCSVM": "N/A", "ECOD": "N/A"}}

    X_scaled = scaler.transform(X_raw)
    X_pca    = pca.transform(X_scaled)

    # Scores continus (anomalie = valeur haute)
    score_if    = float(-models["if"].score_samples(X_pca)[0])
    score_ocsvm = float(-models["ocsvm"].decision_function(X_pca)[0])
    try:
        score_ecod = float(models["ecod"].decision_function(X_pca)[0])
    except Exception:
        score_ecod = 0.0

    # Seuils optimaux (depuis threshold_v3.pkl) ou fallback predict()
    if thresholds:
        thr_if    = thresholds.get("if_threshold",    score_if)
        thr_ocsvm = thresholds.get("ocsvm_threshold", score_ocsvm)
        thr_ecod  = thresholds.get("ecod_threshold",  score_ecod)
        vote_if    = 1 if score_if    >= thr_if    else 0
        vote_ocsvm = 1 if score_ocsvm >= thr_ocsvm else 0
        vote_ecod  = 1 if score_ecod  >= thr_ecod  else 0
    else:
        vote_if    = 1 if models["if"].predict(X_pca)[0]    == -1 else 0
        vote_ocsvm = 1 if models["ocsvm"].predict(X_pca)[0] == -1 else 0
        vote_ecod  = 1 if models["ecod"].predict(X_pca)[0]  ==  1 else 0

    # LOF toujours en binaire (non inclus dans le score continu)
    try:
        vote_lof = 1 if models["lof"].predict(X_pca)[0] == -1 else 0
    except Exception:
        vote_lof = 0

    # Vote majoritaire 2/4 — 2 modèles sur 4 suffisent (meilleur rappel anomalies)
    total      = vote_if + vote_lof + vote_ocsvm + vote_ecod
    is_anomaly = total >= 2           # 2 modeles sur 4 doivent voter ANOMALY
    confidence = round(total / 4.0, 4)  # 0.0 | 0.25 | 0.50 | 0.75 | 1.0

    return {
        "votes":           total,
        "label":           "ANOMALY" if is_anomaly else "NORMAL",
        "confidence":      confidence,
        "is_anomaly":      is_anomaly,
        "individual": {
            "IF":    "ANOMALY" if vote_if    else "NORMAL",
            "LOF":   "ANOMALY" if vote_lof   else "NORMAL",
            "OCSVM": "ANOMALY" if vote_ocsvm else "NORMAL",
            "ECOD":  "ANOMALY" if vote_ecod  else "NORMAL",
        }
    }


def update_history(sensor_id: str, score: float, confidence: float):
    """Enregistre la prédiction dans l'historique glissant du moteur."""
    if sensor_id not in anomaly_history:
        anomaly_history[sensor_id] = deque(maxlen=HISTORY_WINDOW)
    anomaly_history[sensor_id].append({
        "timestamp":  datetime.now().isoformat(),
        "score":      score,
        "confidence": confidence,
    })
    # Calculer la baseline sur les BASELINE_SAMPLES premières mesures
    if sensor_id not in sensor_baseline:
        hist = list(anomaly_history[sensor_id])
        if len(hist) >= BASELINE_SAMPLES:
            scores = [e["score"] for e in hist[:BASELINE_SAMPLES]]
            sensor_baseline[sensor_id] = float(np.mean(scores))
            log.info(f"Baseline capteur {sensor_id} établie : {sensor_baseline[sensor_id]:.4f}")
    # Persistence asynchrone (toutes les 5 min)
    save_history_to_disk()


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIQUE RUL
# ══════════════════════════════════════════════════════════════════════════════

# Seuils industriels roulements (basés sur les données full_data)
# vib_total = √(X²+Y²+Z²) — norme Pythagore 3D
# Seuil vib_total ≈ √3 × seuil_z (car axes équivalents)
VIB_TOTAL_WARN = float(np.sqrt(3) * 600)   # ≈ 1039 mg
VIB_TOTAL_CRIT = float(np.sqrt(3) * 1000)  # ≈ 1732 mg
VIB_TOTAL_MAX  = float(np.sqrt(3) * 1500)  # ≈ 2598 mg

THRESHOLDS = {
    "temp_mean":   {"warn": 50.0,           "crit": 60.0,           "max": 70.0},
    "vib_total":   {"warn": VIB_TOTAL_WARN, "crit": VIB_TOTAL_CRIT, "max": VIB_TOTAL_MAX},
    "vib_z_kurt":  {"warn": 4.0,            "crit": 7.0,            "max": 10.0},
    "vib_z_crest": {"warn": 3.0,            "crit": 5.0,            "max": 8.0},
}

# RUL de référence par niveau de dégradation (heures)
# Seuils alignés sur le cahier des charges :
#   OK        > 14 jours (336h)
#   ATTENTION   7-14 jours (168-336h)
#   URGENT      3-7  jours (72-168h)
#   CRITIQUE  < 3  jours (< 72h)
RUL_TABLE = {
    "OK":        (336, 720),    # 14 à 30 jours
    "ATTENTION": (168, 336),    # 7 à 14 jours
    "URGENT":    (72,  168),    # 3 à 7 jours
    "CRITIQUE":  (0,   72),     # < 3 jours
}


def compute_rul(history: List[MeasurePoint], feat: dict, sensor_id: str, predict_result: dict = None) -> dict:
    """
    Estimation du RUL basée sur :
    1. Score de dégradation actuel (position par rapport aux seuils industriels)
    2. Tendance temporelle des features critiques (pente de régression linéaire)
    3. Historique des scores d'anomalie en mémoire (fenêtre glissante)

    ⚠️  LIMITE CONNUE — FORMULE HEURISTIQUE :
    Ce module utilise une estimation empirique, pas un modèle de régression
    entraîné. Un vrai modèle RUL supervisé (régression de Weibull, LSTM,
    modèle de Cox) nécessite des données de défaillances réelles confirmées
    avec timestamps précis — non disponibles pendant la période de collecte
    (nov. 2025 → mai 2026 : aucun moteur n'a atteint la défaillance complète).

    ⚠️  LIMITE CONNUE — SEUILS UNIFORMES :
    Tous les capteurs sont comparés aux mêmes seuils absolus, sans baseline
    individuelle. Conséquence observée : des capteurs sains (health > 90)
    peuvent recevoir un niveau URGENT à cause de la sensibilité de deg_instant.
    Correction partielle : le seuil FAIBLE a été rehaussé à health_score > 85
    pour filtrer les faux positifs.
    """
    # Filtrage faux positifs : capteurs clairement sains → forcer OK
    health_score_direct = feat.get("health_score", 50.0)
    _force_ok = health_score_direct >= 85.0

    temps     = [h.temperature for h in history if h.temperature is not None]
    vib_x_lst = [h.vibration_x for h in history if h.vibration_x is not None]
    vib_y_lst = [h.vibration_y for h in history if h.vibration_y is not None]
    vib_z_lst = [h.vibration_z for h in history if h.vibration_z is not None]

    # Calcul de la série temporelle vib_total = √(X²+Y²+Z²) par mesure
    n = min(len(vib_x_lst), len(vib_y_lst), len(vib_z_lst))
    vib_total_series = [
        vib_total_pythagorean(
            vib_x_lst[i] if i < len(vib_x_lst) else 0.0,
            vib_y_lst[i] if i < len(vib_y_lst) else 0.0,
            vib_z_lst[i] if i < len(vib_z_lst) else 0.0
        )
        for i in range(n)
    ]
    # Fallback si axes manquants : utilise vib_z seul
    if not vib_total_series and vib_z_lst:
        vib_total_series = vib_z_lst

    # ── 1. Score de dégradation instantané ────────────────────────────────
    deg_scores = []

    for key, thresh in THRESHOLDS.items():
        val = feat.get(key)
        if val is None or np.isnan(val):
            continue
        if val >= thresh["crit"]:
            deg_scores.append(0.85 + 0.15 * norm01(val, thresh["crit"], thresh["max"]))
        elif val >= thresh["warn"]:
            deg_scores.append(0.50 + 0.35 * norm01(val, thresh["warn"], thresh["crit"]))
        else:
            deg_scores.append(norm01(val, 0, thresh["warn"]) * 0.50)

    deg_instant = float(np.mean(deg_scores)) if deg_scores else 0.3

    # ── 2. Taux de dégradation via tendance ────────────────────────────────
    deg_rate = 0.0

    if len(vib_total_series) >= 3:
        slope_vib = safe_trend(vib_total_series)
        # Normaliser la pente par rapport à la plage max vib_total
        deg_rate += max(0.0, slope_vib / (VIB_TOTAL_MAX + 1e-9))

    if len(temps) >= 3:
        slope_temp = safe_trend(temps)
        deg_rate += max(0.0, slope_temp / (THRESHOLDS["temp_mean"]["max"] + 1e-9))

    deg_rate = min(1.0, deg_rate)

    # ── 3. Prise en compte de l'historique anomalies (mémoire moteur) ─────
    hist_factor = 1.0
    if sensor_id in anomaly_history and len(anomaly_history[sensor_id]) >= 5:
        recent_scores = [e["score"] for e in list(anomaly_history[sensor_id])[-10:]]
        anomaly_rate  = sum(1 for s in recent_scores if s >= 0.5) / len(recent_scores)
        # Plus le taux d'anomalies récentes est élevé, plus le RUL est court
        hist_factor = 1.0 - (anomaly_rate * 0.4)

    # ── 4. Score combiné de dégradation ───────────────────────────────────
    deg_combined = (0.50 * deg_instant + 0.30 * deg_rate + 0.20 * (1 - hist_factor))
    deg_combined = min(1.0, max(0.0, deg_combined))

    # ── 5. Niveau d'alerte — cohérent avec risk_level de /v1/predict ──────
    # Correction faux URGENT : capteur sain (health >= 85) → forcer OK
    # Problème observé : la formule deg_instant est trop sensible pour les
    # capteurs avec health 90+, générant des niveaux URGENT non justifiés.
    if _force_ok:
        alert_level = "OK"
        rul_min, rul_max = RUL_TABLE["OK"]
        rul_hours = round(rul_max - (deg_combined / 0.30) * (rul_max - rul_min), 1)
        rul_hours = max(336.0, rul_hours)   # Plancher 336h (14 jours) pour capteurs sains — seuil CDC
        rul_days  = round(rul_hours / 24.0, 2)
        n_pts = len(history)
        confidence = "HAUTE" if n_pts >= 10 else ("MOYENNE" if n_pts >= 5 else "FAIBLE")
        return {
            "rul_hours":        rul_hours,
            "rul_days":         rul_days,
            "degradation_rate": round(deg_combined * 100, 2),
            "health_score":     health_score_direct,
            "confidence":       confidence,
            "alert_level":      "OK",
            "recommendation":   "Fonctionnement normal. Capteur sain (health >= 85). Prochaine inspection planifiée.",
            "trend":            {
                "temp_trend":          round(safe_trend([h.temperature for h in history if h.temperature is not None]), 4),
                "vib_total_trend":     0.0,
                "vib_formula":        "sqrt(X2 + Y2 + Z2)",
                "deg_instant":        round(deg_instant, 4),
                "deg_rate":           round(deg_rate, 4),
                "hist_anomaly_factor":0.0,
                "note":              "Niveau force a OK — health_score >= 85 (filtre faux positifs)",
            }
        }

    # On utilise le risk_level de la prédiction comme référence principale
    predict_risk = (predict_result or {}).get("risk_level", "")
    if predict_risk == "CRITIQUE":
        # CRITIQUE ML : alerte selon deg_combined pour nuancer
        if deg_combined >= 0.60:
            alert_level = "CRITIQUE"
        else:
            alert_level = "URGENT"
    elif predict_risk in ("ÉLEVÉ", "MODÉRÉ"):
        if deg_combined >= 0.55:
            alert_level = "URGENT"
        else:
            alert_level = "ATTENTION"
    elif predict_risk == "FAIBLE":
        # Même si ML dit FAIBLE, surveiller si dégradation physique élevée
        if deg_combined >= 0.80:
            alert_level = "URGENT"
        elif deg_combined >= 0.30:
            alert_level = "ATTENTION"
        else:
            alert_level = "OK"
    else:
        # Fallback sur deg_combined seul
        if deg_combined >= 0.80:
            alert_level = "CRITIQUE"
        elif deg_combined >= 0.55:
            alert_level = "URGENT"
        elif deg_combined >= 0.30:
            alert_level = "ATTENTION"
        else:
            alert_level = "OK"

    # ── 6. Estimation RUL en heures ───────────────────────────────────────
    rul_min, rul_max = RUL_TABLE[alert_level]
    # Interpolation linéaire dans la plage de l'alerte
    if alert_level == "OK":
        rul_hours = rul_max - (deg_combined / 0.30) * (rul_max - rul_min)
    elif alert_level == "ATTENTION":
        ratio = (deg_combined - 0.30) / 0.25
        rul_hours = rul_max - ratio * (rul_max - rul_min)
    elif alert_level == "URGENT":
        ratio = (deg_combined - 0.55) / 0.25
        rul_hours = rul_max - ratio * (rul_max - rul_min)
    else:  # CRITIQUE
        ratio = (deg_combined - 0.80) / 0.20
        rul_hours = max(0.0, rul_max - ratio * rul_max)

    rul_hours = round(max(0.0, rul_hours), 1)
    rul_days  = round(rul_hours / 24.0, 2)

    # ── 7. Confiance de l'estimation ──────────────────────────────────────
    n_pts = len(history)
    if n_pts >= 10:
        confidence = "HAUTE"
    elif n_pts >= 5:
        confidence = "MOYENNE"
    else:
        confidence = "FAIBLE"

    # ── 8. Recommandation ────────────────────────────────────────────────
    recommendations = {
        "OK":        f"Fonctionnement normal. RUL > 14 jours. Prochaine inspection planifiée selon calendrier.",
        "ATTENTION": f"Surveillance renforcée. RUL 7-14 jours. Planifier une inspection préventive sous 7 jours.",
        "URGENT":    f"Intervention requise. RUL 3-7 jours. Commander les pièces et programmer la maintenance sous 3 jours.",
        "CRITIQUE":  f"ARRÊT IMMÉDIAT recommandé. RUL < 3 jours. Risque de défaillance imminente du roulement.",
    }

    # ── 9. Tendances par feature ──────────────────────────────────────────
    trend_detail = {
        "temp_trend":        round(safe_trend(temps), 4),
        "vib_total_trend":   round(safe_trend(vib_total_series), 4),
        "vib_total_current": round(vib_total_series[-1], 2) if vib_total_series else 0.0,
        "vib_formula":       "sqrt(X² + Y² + Z²)",
        "deg_instant":       round(deg_instant, 4),
        "deg_rate":          round(deg_rate, 4),
        "hist_anomaly_factor": round(1 - hist_factor, 4),
    }

    return {
        "rul_hours":        rul_hours,
        "rul_days":         rul_days,
        "degradation_rate": round(deg_combined * 100, 2),
        "health_score":     feat.get("health_score", 50.0),
        "confidence":       confidence,
        "alert_level":      alert_level,
        "recommendation":   recommendations[alert_level],
        "trend":            trend_detail,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

if FASTAPI_OK:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Remplace @app.on_event('startup') — recommandé depuis FastAPI 0.93."""
        log.info(f"Démarrage API Unifiée V{API_VERSION}")
        load_all_models()
        load_history_from_disk()
        yield
        # Shutdown : sauvegarder une dernière fois avant arrêt
        import time as _t
        global _last_persist
        _last_persist = 0.0
        save_history_to_disk()
        log.info("API arrêtée proprement — historique sauvegardé")

    app = FastAPI(
        title="Maintenance Prédictive — API Unifiée",
        description=(
            "Système complet de surveillance de 20 capteurs IFM — Novation City.\n\n"
            "**Modèle IA** : Ensemble non supervisé IF + LOF + OCSVM + ECOD (vote 2/4)\n\n"
            "**Données** : Capteurs IFM VVB001 → MySQL ai_cp (1 648 886 mesures, nov 2025 – mar 2026)\n\n"
            "**PFE ISG Bizerte** — Détection d'anomalies + Estimation RUL roulements"
        ),
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Accueil ────────────────────────────────────────────────────────────
    @app.get("/", tags=["Système"])
    def root():
        return {
            "api":     "Maintenance Prédictive — API Unifiée",
            "version": API_VERSION,
            "port":    8000,
            "docs":    "http://localhost:8000/docs",
            "endpoints": {
                "POST /v1/predict":               "Détection anomalie temps réel (IF+LOF+OCSVM+ECOD)",
                "POST /v1/predict-rul":           "Estimation RUL (Remaining Useful Life)",
                "POST /v1/iot-predict":           "Prédiction directe IoT sans base de données [NEW]",
                "GET  /v1/health-score/{id}":     "Score santé moteur (0-100)",
                "GET  /v1/history/{id}":          "Historique prédictions par capteur",
                "GET  /v1/alert-level/{id}":      "Niveau alerte actuel — dashboard",
                "GET  /health":                   "Health check API",
                "GET  /metrics":                  "Métriques modèle (F1=0.401, AUC=0.683)",
                "GET  /sensors":                  "Liste 20 capteurs IFM",
                "GET  /anomalies":                "Anomalies filtrées par score",
            }
        }

    # ── Health Check ───────────────────────────────────────────────────────
    @app.get("/health", tags=["Système"])
    def health():
        return {
            "status":         "ok",
            "models_loaded":  len(models) == 4,
            "models":         list(models.keys()),
            "features_count": len(features_list),
            "version":        API_VERSION,
            "n_sensors_in_memory": len(anomaly_history),
            "timestamp":      datetime.now().isoformat(),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  POST /v1/predict — Détection anomalie
    # ══════════════════════════════════════════════════════════════════════
    @app.post(
        "/v1/predict",
        response_model=PredictResponse,
        tags=["IA / Prédiction"],
        summary="Détection d'anomalie temps réel",
        description=(
            "Reçoit l'historique de mesures d'un capteur et retourne le diagnostic "
            "d'anomalie basé sur l'ensemble IF + LOF + OCSVM + ECOD (vote majoritaire 2/4).\n\n"
            "**Format données** : compatible avec le champ `data` de la table `full_data` "
            "(SensorNodeId, Temperature, Vibration RMS X/Y/Z)."
        )
    )
    def predict_anomaly(req: PredictRequest):
        if not req.history:
            raise HTTPException(status_code=400, detail="history ne peut pas être vide")

        # 1. Extraction features
        feat = extract_features(req.history)

        # 2. Construction vecteur pour le modèle
        if features_list:
            X = np.array([[feat.get(c, np.nan) for c in features_list]], dtype="float32")
            X = np.nan_to_num(X, nan=0.0)
        else:
            # Fallback : vecteur par défaut si features non chargées
            X = np.array([[
                feat.get("temp_mean", 35),
                feat.get("vib_z_rms_w", 300),
                feat.get("vib_z_kurt", 0),
                feat.get("vib_x_rms_w", 200),
                feat.get("vib_y_rms_w", 200),
            ]], dtype="float32")

        # 3. Inférence ensemble
        result = run_ensemble(X)

        # 4. Calcul anomaly_score normalisé (0–1)
        anomaly_score = round(result["confidence"], 4)
        # V6 : boost contextuel base sur les valeurs physiques reelles
        if result["is_anomaly"]:
            vib_rms = feat.get("vib_z_rms_w", 0) or 0
            if vib_rms > 1000:
                anomaly_score = min(1.0, anomaly_score + 0.15)
            elif vib_rms > 600:
                anomaly_score = min(1.0, anomaly_score + 0.05)

        # 5. Niveau de risque
        if anomaly_score >= 0.75:   risk = "CRITIQUE"
        elif anomaly_score >= 0.50: risk = "ÉLEVÉ"
        elif anomaly_score >= 0.25: risk = "MODÉRÉ"
        else:                       risk = "FAIBLE"

        # Cohérence health/risk : capteur sain (health ≥ 85) non confirmé anomalie → FAIBLE
        # Évite MODÉRÉ+Health=99 qui est contradictoire pour l'utilisateur
        _health_val = feat.get("health_score", 0) or 0
        if _health_val >= 85 and not result["is_anomaly"]:
            risk = "FAIBLE"

        # 6. Mise à jour historique moteur (pour RUL)
        update_history(req.sensor_id, anomaly_score, result["confidence"])

        # 6b. Envoi alerte externe (email / webhook / SMS) si niveau critique
        if ALERTS_ENABLED and _alert_manager and result["is_anomaly"] and risk in ("CRITIQUE", "ÉLEVÉ"):
            _alert_manager.send_alert(
                sensor_id   = req.sensor_id,
                risk_level  = risk,
                health_score= feat.get("health_score", 0),
                rul_hours   = None,   # RUL calculé séparément dans /v1/predict-rul
                vib_total   = feat.get("vib_total"),
                temperature = feat.get("temp_cur"),
                votes       = result["votes"]
            )

        # 7. Features utiles à retourner (pour debug / dashboard)
        feat_summary = {
            k: round(v, 4) if isinstance(v, float) and not np.isnan(v) else v
            for k, v in feat.items()
        }

        return PredictResponse(
            sensor_id         = req.sensor_id,
            motor_id          = req.motor_id,
            timestamp         = datetime.now().isoformat(),
            prediction        = result["label"],
            is_anomaly        = result["is_anomaly"],
            confidence        = result["confidence"],
            votes             = result["votes"],
            risk_level        = risk,
            anomaly_score     = anomaly_score,
            individual_models = result["individual"],
            features          = feat_summary,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  POST /v1/predict-rul — Remaining Useful Life
    # ══════════════════════════════════════════════════════════════════════
    @app.post(
        "/v1/predict-rul",
        response_model=RULResponse,
        tags=["IA / Prédiction"],
        summary="Estimation du Remaining Useful Life (RUL)",
        description=(
            "Estime le temps restant avant défaillance du roulement en heures.\n\n"
            "**Méthode** :\n"
            "- Score de dégradation instantané (position par rapport aux seuils industriels)\n"
            "- Tendance temporelle des vibrations et de la température (régression linéaire)\n"
            "- Historique des anomalies précédentes du moteur (fenêtre glissante)\n\n"
            "**Minimum requis** : 3 mesures dans `history` pour calculer la tendance.\n\n"
            "**Seuils utilisés** (roulements industriels) :\n"
            "- Température critique : > 60°C\n"
            "- Vibration Z RMS critique : > 1000 mg\n"
            "- Kurtosis critique : > 7"
        )
    )
    def predict_rul(req: RULRequest):
        if len(req.history) < 3:
            raise HTTPException(
                status_code=400,
                detail="Minimum 3 mesures requises pour estimer le RUL (calcul de tendance)"
            )

        # 1. Extraction features temporelles
        feat = extract_features(req.history)

        # 2. Enrichissement features spectrales (si signal_processing disponible)
        if SIGNAL_PROCESSING_OK:
            try:
                vib_series = [h.vibration_z for h in req.history if h.vibration_z is not None]
                if len(vib_series) >= 8:
                    spec_feat = extract_spectral_features(vib_series, fs=100.0, rpm=1450.0)
                    feat.update(spec_feat)
            except Exception as _e:
                log.debug(f"Spectral features RUL ignorées : {_e}")

        # 3. Essai modèle RUL ML dédié (GradientBoosting)
        ml_rul_result = None
        if RUL_ML_ENABLED and _rul_predictor is not None:
            try:
                ml_rul_result = _rul_predictor.predict(feat)
                log.debug(f"RUL ML : {ml_rul_result['rul_hours']}h ({ml_rul_result['model_type']})")
            except Exception as _e:
                log.warning(f"RUL ML échoué, fallback heuristique : {_e}")

        # 4. Calcul RUL heuristique (toujours calculé pour les tendances)
        predict_result_data = {
            "prediction":    req.prediction    or "NORMAL",
            "votes":         req.votes         or 0,
            "confidence":    req.confidence    or 0.0,
            "risk_level":    req.risk_level     or "OK",
            "anomaly_score": req.anomaly_score or 0.0,
        }
        rul_heuristic = compute_rul(req.history, feat, req.sensor_id, predict_result_data)

        # 5. Sélection du résultat final : ML si disponible et cohérent, heuristique sinon
        # Règle de cohérence : si ML dit "OK"/"ATTENTION" mais anomalie est CRITIQUE/ÉLEVÉ
        # → le modèle ML synthétique n'est pas fiable dans ce cas, on garde l'heuristique
        _predict_risk = (req.risk_level or "OK").upper()
        _ml_alert     = (ml_rul_result or {}).get("alert_level", "OK")
        _ml_incoherent = (
            _predict_risk in ("CRITIQUE", "ÉLEVÉ") and
            _ml_alert in ("OK", "ATTENTION")
        )
        if ml_rul_result and ml_rul_result.get("model_type") != "heuristic_fallback" and not _ml_incoherent:
            rul_hours    = ml_rul_result["rul_hours"]
            rul_days     = ml_rul_result["rul_days"]
            confidence   = ml_rul_result["confidence"]
            alert_level  = ml_rul_result["alert_level"]
            recommendation = ml_rul_result["recommendation"]
            trend_detail = rul_heuristic["trend"]
            trend_detail["rul_model"] = ml_rul_result["model_type"]
        else:
            rul_hours    = rul_heuristic["rul_hours"]
            rul_days     = rul_heuristic["rul_days"]
            confidence   = rul_heuristic["confidence"]
            alert_level  = rul_heuristic["alert_level"]
            recommendation = rul_heuristic["recommendation"]
            trend_detail = rul_heuristic["trend"]
            trend_detail["rul_model"] = "heuristic_v6"

        # 6. Mise à jour historique
        deg_score = rul_heuristic["degradation_rate"] / 100.0
        update_history(req.sensor_id, deg_score, 1.0)

        # 7. Alerte externe si RUL sous seuil CDC (URGENT < 7j, CRITIQUE < 3j)
        if ALERTS_ENABLED and _alert_manager and alert_level in ("URGENT", "CRITIQUE"):
            _alert_manager.send_alert(
                sensor_id   = req.sensor_id,
                risk_level  = alert_level,
                health_score= rul_heuristic["health_score"],
                rul_hours   = rul_hours,
                vib_total   = None,
                temperature = None,
                votes       = 0
            )

        return RULResponse(
            sensor_id        = req.sensor_id,
            motor_id         = req.motor_id,
            timestamp        = datetime.now().isoformat(),
            rul_hours        = rul_hours,
            rul_days         = rul_days,
            degradation_rate = rul_heuristic["degradation_rate"],
            health_score     = rul_heuristic["health_score"],
            confidence       = confidence,
            alert_level      = alert_level,
            recommendation   = recommendation,
            trend            = trend_detail,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  POST /v1/iot-predict — Predict sans base de données (IoT direct)
    # ══════════════════════════════════════════════════════════════════════
    @app.post(
        "/v1/iot-predict",
        response_model=IoTPredictResponse,
        tags=["IA / Prédiction"],
        summary="Prédiction directe depuis données IoT (sans base de données)",
        description=(
            "Endpoint destiné à la **production sans accès MariaDB**.\n\n"
            "Le collègue qui reçoit les données IoT envoie **une mesure à la fois** "
            "(température + vibration X/Y/Z consolidées). "
            "Le serveur maintient une **fenêtre glissante de 10 mesures** par capteur "
            "et retourne la prédiction d'anomalie **ET** le RUL en un seul appel.\n\n"
            "**Format d'entrée** : mesure consolidée issue des 3 lignes `full_data` "
            "(`gph='temperature'` + `gph='vibration_x'` + `gph='vibration_y'`).\n\n"
            "**Exemple d'utilisation** :\n"
            "```\n"
            "POST /v1/iot-predict\n"
            "{\n"
            '  "sensor_id": "8f7f2f7e",\n'
            '  "temperature": 32.5,\n'
            '  "vibration_x": 266.0,\n'
            '  "vibration_y": 273.0,\n'
            '  "vibration_z": 280.0\n'
            "}\n"
            "```"
        )
    )
    def iot_predict(req: IoTMeasurementRequest):
        global iot_windows

        # 1. Construire un MeasurePoint depuis la mesure brute
        point = MeasurePoint(
            timestamp   = req.timestamp or datetime.now().isoformat(),
            temperature = req.temperature,
            vibration_x = req.vibration_x,
            vibration_y = req.vibration_y,
            vibration_z = req.vibration_z,
            current     = req.current or 0.0,
            acc_p2p     = req.acc_p2p,
            acc_rms     = req.acc_rms,
            acc_crest   = req.acc_crest,
            acc_z2p     = req.acc_z2p,
        )

        # 2. Ajouter à la fenêtre glissante serveur
        if req.sensor_id not in iot_windows:
            iot_windows[req.sensor_id] = deque(maxlen=IOT_WINDOW_SIZE)
        iot_windows[req.sensor_id].append(point)
        history = list(iot_windows[req.sensor_id])
        window_size = len(history)

        # 3. Extraction features
        feat = extract_features(history)

        # 4. Construction vecteur pour les modèles
        if features_list:
            X = np.array([[feat.get(c, np.nan) for c in features_list]], dtype="float32")
            X = np.nan_to_num(X, nan=0.0)
        else:
            X = np.array([[
                feat.get("temp_mean", 35),
                feat.get("vib_z_rms_w", 300),
                feat.get("vib_z_kurt", 0),
                feat.get("vib_x_rms_w", 200),
                feat.get("vib_y_rms_w", 200),
            ]], dtype="float32")

        # 5. Inférence ensemble
        result = run_ensemble(X)

        # 6. Calcul anomaly_score
        anomaly_score = round(result["confidence"], 4)
        if result["is_anomaly"]:
            vib_rms = feat.get("vib_z_rms_w", 0) or 0
            if vib_rms > 1000:
                anomaly_score = min(1.0, anomaly_score + 0.15)
            elif vib_rms > 600:
                anomaly_score = min(1.0, anomaly_score + 0.05)

        # 7. Niveau de risque
        if anomaly_score >= 0.75:   risk = "CRITIQUE"
        elif anomaly_score >= 0.50: risk = "ÉLEVÉ"
        elif anomaly_score >= 0.25: risk = "MODÉRÉ"
        else:                       risk = "FAIBLE"

        _health_val = feat.get("health_score", 0) or 0
        if _health_val >= 85 and not result["is_anomaly"]:
            risk = "FAIBLE"

        # 8. Mise à jour historique anomalies
        update_history(req.sensor_id, anomaly_score, result["confidence"])

        # 9. Alerte externe si CRITIQUE / ÉLEVÉ
        if ALERTS_ENABLED and _alert_manager and result["is_anomaly"] and risk in ("CRITIQUE", "ÉLEVÉ"):
            _alert_manager.send_alert(
                sensor_id    = req.sensor_id,
                risk_level   = risk,
                health_score = feat.get("health_score", 0),
                rul_hours    = None,
                vib_total    = feat.get("vib_total"),
                temperature  = feat.get("temp_cur"),
                votes        = result["votes"]
            )

        # 10. RUL — calculé uniquement si >= 3 mesures disponibles
        rul_hours = rul_days = health_score = alert_level = recommendation = None
        if window_size >= 3:
            predict_result_data = {
                "prediction":    result["label"],
                "votes":         result["votes"],
                "confidence":    result["confidence"],
                "risk_level":    risk,
                "anomaly_score": anomaly_score,
            }
            try:
                rul_result = compute_rul(history, feat, req.sensor_id, predict_result_data)
                rul_hours      = rul_result["rul_hours"]
                rul_days       = rul_result["rul_days"]
                health_score   = rul_result["health_score"]
                alert_level    = rul_result["alert_level"]
                recommendation = rul_result["recommendation"]
            except Exception as _rul_err:
                log.warning(f"RUL IoT échoué pour {req.sensor_id} : {_rul_err}")

        # 11. Features résumées
        feat_summary = {
            k: round(v, 4) if isinstance(v, float) and not np.isnan(v) else v
            for k, v in feat.items()
        }

        return IoTPredictResponse(
            sensor_id         = req.sensor_id,
            motor_id          = req.motor_id,
            timestamp         = datetime.now().isoformat(),
            window_size       = window_size,
            prediction        = result["label"],
            is_anomaly        = result["is_anomaly"],
            confidence        = result["confidence"],
            votes             = result["votes"],
            risk_level        = risk,
            anomaly_score     = anomaly_score,
            individual_models = result["individual"],
            rul_hours         = rul_hours,
            rul_days          = rul_days,
            health_score      = health_score,
            alert_level       = alert_level,
            recommendation    = recommendation,
            features          = feat_summary,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  GET /v1/health-score/{sensor_id}
    # ══════════════════════════════════════════════════════════════════════
    @app.get(
        "/v1/health-score/{sensor_id}",
        tags=["IA / Prédiction"],
        summary="Score de santé d'un moteur",
    )
    def get_health_score(sensor_id: str):
        """
        Retourne le score de santé (0–100) normalisé par capteur.
        Utilise la baseline propre au capteur pour éviter le biais global.
        """
        if sensor_id not in anomaly_history or not anomaly_history[sensor_id]:
            return {
                "sensor_id":    sensor_id,
                "health_score": 100.0,
                "status":       "Aucun historique disponible pour ce capteur",
                "n_records":    0,
            }

        hist   = list(anomaly_history[sensor_id])
        scores = [e["score"] for e in hist]
        recent = scores[-10:]

        # Score brut
        raw_health = 100 * (1 - np.mean(recent))

        # Normalisation par baseline capteur — corrige le biais global 43-48
        # Si baseline connue : on recentre le score autour de 100 (baseline = 0% dégradation)
        baseline = sensor_baseline.get(sensor_id)
        if baseline is not None and baseline > 0:
            # Score relatif : combien on a dégradé par rapport à la baseline
            degradation_relative = max(0.0, np.mean(recent) - baseline)
            health = round(100 * (1 - degradation_relative / max(baseline, 0.01)), 1)
            health = max(0.0, min(100.0, health))
            score_method = "relatif_baseline"
        else:
            health = round(max(0.0, min(100.0, raw_health)), 1)
            score_method = "brut_en_attente_baseline"

        anomaly_rate = round(sum(1 for s in recent if s >= 0.5) / len(recent), 3)
        trend_val    = safe_trend(scores[-5:]) if len(scores) >= 5 else 0.0

        return {
            "sensor_id":     sensor_id,
            "health_score":  health,
            "health_raw":    round(raw_health, 1),
            "baseline":      round(baseline, 4) if baseline else None,
            "score_method":  score_method,
            "anomaly_rate":  anomaly_rate,
            "n_records":     len(hist),
            "last_score":    round(scores[-1], 4),
            "trend":         "DÉGRADATION" if trend_val > 0.01 else (
                             "AMÉLIORATION" if trend_val < -0.01 else "STABLE"),
            "timestamp":     hist[-1]["timestamp"],
        }

    # ── Métriques modèle ───────────────────────────────────────────────────
    @app.get("/metrics", tags=["Système"],
             summary="Métriques du modèle V3 (F1, AUC, Accuracy)")
    def get_metrics():
        """
        Retourne les métriques de performance du modèle non supervisé.
        Source : models/metrics_v3.csv (F1=0.4008, AUC=0.6830, Acc=0.9232)
        """
        path = Path(METRICS_PATH)
        if not path.exists():
            # Chercher dans tous les emplacements possibles
            for candidate in [
                MODEL_DIR / "metrics_v2.csv",
                MODEL_DIR / "metrics_v3.csv",
                PROJECT_DIR / "metrics_v2.csv",
            ]:
                if candidate.exists():
                    path = candidate
                    break
            else:
                return {
                    "message": "Métriques non disponibles — lance step3_model.py",
                    "hint":    "Le fichier metrics_v2.csv doit être dans le dossier models/",
                }
        try:
            # latin-1 accepte tous les octets 0-255 — toujours safe sur Windows
            df_m = pd.read_csv(path, encoding="latin-1")
            # Format long (metric,value) → dict {metric_name: value}
            if "metric" in df_m.columns and "value" in df_m.columns:
                m = df_m.set_index("metric")["value"].to_dict()
            else:
                m = df_m.iloc[0].to_dict()
            return {
                "model_version": "V3",
                "ensemble":      "IF + LOF + OCSVM + ECOD",
                "voting":        "2/4 (majoritaire)",
                "dataset":       "ai_cp full_data — 79940 sessions, 20 capteurs IFM, nov2025-mar2026",
                "f1_score":      round(float(m.get("f1_score",  0)), 4),
                "accuracy":      round(float(m.get("accuracy",  0)), 4),
                "precision":     round(float(m.get("precision", 0)), 4),
                "recall":        round(float(m.get("recall",    0)), 4),
                "auc_roc":       round(float(m.get("auc_roc",   0)), 4),
                "n_anomalies":   int(float(m.get("n_anomalies", 0))),
                "n_total":       int(float(m.get("n_total",     0))),
                "contamination": round(float(m.get("contamination", 0)), 4),
                "weights": {
                    "IF":    round(float(m.get("weights_if",    0.2)), 2),
                    "LOF":   round(float(m.get("weights_lof",   0.3)), 2),
                    "OCSVM": round(float(m.get("weights_ocsvm", 0.5)), 2),
                },
                "source_file": str(path.name),
            }
        except Exception as e:
            return {"message": f"Erreur lecture métriques : {e}"}

    # ══════════════════════════════════════════════════════════════════════
    #  POST /v1/spectral-analysis — Analyse spectrale FFT + défauts roulements
    # ══════════════════════════════════════════════════════════════════════
    @app.post(
        "/v1/spectral-analysis",
        tags=["IA / Prédiction"],
        summary="Analyse spectrale FFT et détection de défauts de roulements",
        description=(
            "Effectue une analyse complète du signal de vibration :\n\n"
            "- **FFT** : spectre de puissance, fréquences dominantes, énergie par bande\n"
            "- **Analyse d'enveloppe** : démodulation Hilbert, détection défauts roulements\n"
            "- **Fréquences caractéristiques** : BPFO, BPFI, BSF, FTF (SKF 6205-2RS)\n"
            "- **Ondelettes** : décomposition CWT Morlet pour transitoires\n\n"
            "**Prérequis** : signal_processing.py installé (scipy requis)"
        )
    )
    def spectral_analysis(req: PredictRequest, rpm: float = 1450.0, fs: float = 100.0):
        if not SIGNAL_PROCESSING_OK:
            raise HTTPException(
                status_code=503,
                detail="Module signal_processing non disponible. Vérifier l'installation de scipy."
            )

        vib_series = [h.vibration_z for h in req.history if h.vibration_z is not None]
        if len(vib_series) < 8:
            raise HTTPException(
                status_code=400,
                detail=f"Minimum 8 mesures de vibration_z requises (reçu : {len(vib_series)})"
            )

        try:
            result = full_signal_pipeline(
                vib_signal=vib_series,
                fs=fs,
                rpm=rpm,
                include_raw_spectra=False
            )

            # Enrichir avec features vectorisées pour le ML
            spec_feat = extract_spectral_features(vib_series, fs=fs, rpm=rpm)

            return {
                "sensor_id":         req.sensor_id,
                "timestamp":         datetime.now().isoformat(),
                "signal_length":     len(vib_series),
                "analysis_params":   {"fs_hz": fs, "rpm": rpm},
                "spectral_features": result["spectral_features"],
                "bearing_analysis":  result["bearing_analysis"],
                "wavelet":           result["wavelet"],
                "metadata":          result["metadata"],
                "ml_feature_vector": spec_feat,
            }
        except Exception as e:
            log.error(f"Erreur analyse spectrale : {e}")
            raise HTTPException(status_code=500, detail=f"Erreur analyse : {str(e)}")

    # ══════════════════════════════════════════════════════════════════════
    #  GET /v1/report — Génération de rapport de maintenance HTML/JSON
    # ══════════════════════════════════════════════════════════════════════
    @app.get(
        "/v1/report",
        tags=["Reporting"],
        summary="Génère un rapport de maintenance",
        description=(
            "Génère un rapport de maintenance à partir des données temps réel.\n\n"
            "- **format=html** : Rapport HTML complet (KPIs, planning, capteurs)\n"
            "- **format=json** : Rapport JSON pour intégration\n"
            "- **type** : `daily` (24h) | `weekly` (7j) | `monthly` (30j) | `full`"
        )
    )
    def get_report(
        type: str = "daily",
        format: str = "json",
        sensor_id: Optional[str] = None
    ):
        if not REPORTING_OK:
            raise HTTPException(
                status_code=503,
                detail="Module reporting_module non disponible."
            )
        if type not in ("daily", "weekly", "monthly", "full"):
            raise HTTPException(status_code=400, detail="type doit être : daily | weekly | monthly | full")

        try:
            if format == "html":
                from fastapi.responses import HTMLResponse
                html = generate_html_report(report_type=type, sensor_filter=sensor_id)
                # Optionnel : sauvegarder le rapport
                try:
                    save_report(html, report_type=type)
                except Exception:
                    pass
                return HTMLResponse(content=html)
            else:
                return generate_json_report(report_type=type)
        except Exception as e:
            log.error(f"Erreur génération rapport : {e}")
            raise HTTPException(status_code=500, detail=f"Erreur rapport : {str(e)}")

    # ── Liste capteurs ────────────────────────────────────────────────────
    @app.get("/sensors", tags=["Données"])
    def get_sensors():
        # ── Priorité 1 : anomaly_history temps réel (rempli par /v1/predict) ──
        if anomaly_history:
            try:
                sensors_list = []
                for sid, dq in anomaly_history.items():
                    hist = list(dq)
                    if not hist:
                        continue
                    scores = [e.get("score", 0) for e in hist]
                    n_anom = sum(1 for s in scores if s >= 0.5)
                    avg_s  = round(sum(scores) / len(scores), 3)
                    sensors_list.append({
                        "sensor_id":    sid,
                        "n_measures":   len(hist),
                        "n_anomalies":  n_anom,
                        "anomaly_rate": round(n_anom / len(hist), 3),
                        "avg_score":    avg_s,
                        "avg_health":   round(max(0, 100 - avg_s * 100), 1),
                    })
                if sensors_list:
                    return {"sensors": sensors_list, "source": "realtime"}
            except Exception as e:
                log.warning(f"/sensors fallback erreur : {e}")

        # ── Priorité 2 : df_results historique (fichier CSV pré-calculé) ──
        if df_results is not None:
            try:
                summary = (
                    df_results.groupby("sensor_id")
                    .agg(
                        n_measures  =("is_anomaly", "count"),
                        n_anomalies =("is_anomaly", "sum"),
                        avg_score   =("anomaly_score", "mean"),
                        avg_health  =("health_score", "mean"),
                    )
                    .reset_index()
                )
                summary["anomaly_rate"] = (
                    summary["n_anomalies"] / summary["n_measures"]
                ).round(3)
                return {"sensors": summary.to_dict(orient="records"), "source": "historical"}
            except Exception as e:
                return {"sensors": [], "error": str(e)}

        return {"sensors": [], "message": "Aucune donnée — lance realtime_mariadb.py"}

    # ══════════════════════════════════════════════════════════════════════
    #  GET /v1/history/{sensor_id} — Historique prédictions [NEW]
    # ══════════════════════════════════════════════════════════════════════
    @app.get(
        "/v1/history/{sensor_id}",
        tags=["IA / Prédiction"],
        summary="Historique des prédictions d'un capteur",
        description=(
            "Retourne les N dernières prédictions enregistrées en mémoire "
            "pour un capteur donné. Utile pour visualiser la tendance de dégradation "
            "dans un dashboard ou pour debug.\n\n"
            "**Note** : L'historique est en RAM — réinitialisé au redémarrage de l'API."
        )
    )
    def get_history(sensor_id: str, limit: int = 20):
        """
        Retourne l'historique glissant des scores d'anomalie pour un capteur.
        Maximum HISTORY_WINDOW (50) entrées gardées en mémoire.
        """
        if sensor_id not in anomaly_history or not anomaly_history[sensor_id]:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun historique pour le capteur '{sensor_id}'. "
                       f"Lance d'abord POST /v1/predict avec ce sensor_id."
            )

        hist = list(anomaly_history[sensor_id])
        hist_limited = hist[-limit:]  # Garder les N plus récents

        scores = [e["score"] for e in hist]
        recent = scores[-10:]

        # Tendance : en hausse = dégradation, en baisse = amélioration
        trend_val = safe_trend(scores[-5:]) if len(scores) >= 5 else 0.0
        if trend_val > 0.02:
            trend_label = "DÉGRADATION"
        elif trend_val < -0.02:
            trend_label = "AMÉLIORATION"
        else:
            trend_label = "STABLE"

        return {
            "sensor_id":      sensor_id,
            "n_total":        len(hist),
            "n_returned":     len(hist_limited),
            "avg_score":      round(float(np.mean(recent)), 4),
            "max_score":      round(float(np.max(scores)), 4),
            "anomaly_rate":   round(sum(1 for s in recent if s >= 0.5) / len(recent), 3),
            "trend":          trend_label,
            "trend_value":    round(trend_val, 6),
            "history":        hist_limited,
            "timestamp":      datetime.now().isoformat(),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  GET /v1/alert-level/{sensor_id} — Niveau d'alerte actuel [NEW]
    # ══════════════════════════════════════════════════════════════════════
    @app.get(
        "/v1/alert-level/{sensor_id}",
        tags=["IA / Prédiction"],
        summary="Niveau d'alerte actuel d'un capteur",
        description=(
            "Retourne le niveau d'alerte consolidé d'un capteur basé sur "
            "ses 10 dernières prédictions en mémoire.\n\n"
            "**Niveaux** : OK → ATTENTION → URGENT → CRITIQUE\n\n"
            "Conçu pour alimenter des feux tricolores dans un dashboard."
        )
    )
    def get_alert_level(sensor_id: str):
        """
        Calcule le niveau d'alerte consolidé à partir de l'historique en mémoire.
        Retourne une réponse simple pour dashboard / feux tricolores.
        """
        if sensor_id not in anomaly_history or not anomaly_history[sensor_id]:
            return {
                "sensor_id":   sensor_id,
                "alert_level": "INCONNU",
                "color":       "gray",
                "message":     "Aucune prédiction reçue pour ce capteur.",
                "timestamp":   datetime.now().isoformat(),
            }

        hist   = list(anomaly_history[sensor_id])
        scores = [e["score"] for e in hist[-10:]]  # 10 dernières
        avg    = float(np.mean(scores))
        trend  = safe_trend(scores) if len(scores) >= 3 else 0.0
        anomaly_rate = sum(1 for s in scores if s >= 0.5) / len(scores)

        # Calcul niveau d'alerte consolidé
        if avg >= 0.75 or anomaly_rate >= 0.80:
            alert, color, icon = "CRITIQUE",  "red",    "🔴"
        elif avg >= 0.50 or anomaly_rate >= 0.50:
            alert, color, icon = "URGENT",    "orange", "🟠"
        elif avg >= 0.25 or anomaly_rate >= 0.20:
            alert, color, icon = "ATTENTION", "yellow", "🟡"
        else:
            alert, color, icon = "OK",        "green",  "🟢"

        messages = {
            "OK":        "Fonctionnement nominal.",
            "ATTENTION": "Surveillance renforcée recommandée.",
            "URGENT":    "Intervention recommandée sous 72h.",
            "CRITIQUE":  "ARRÊT IMMÉDIAT recommandé.",
        }

        return {
            "sensor_id":    sensor_id,
            "alert_level":  alert,
            "color":        color,
            "icon":         icon,
            "avg_score":    round(avg, 4),
            "anomaly_rate": round(anomaly_rate, 3),
            "trend":        "↑ HAUSSE" if trend > 0.02 else ("↓ BAISSE" if trend < -0.02 else "→ STABLE"),
            "n_measures":   len(scores),
            "message":      messages[alert],
            "timestamp":    datetime.now().isoformat(),
        }

    # ── Anomalies filtrées ────────────────────────────────────────────────
    @app.get("/anomalies", tags=["Données"])
    def get_anomalies(min_score: float = 0.5, limit: int = 100):
        if df_results is None:
            return {"anomalies": []}
        df_a = df_results[df_results["anomaly_score"] >= min_score]
        cols = [c for c in ["sensor_id", "motor_id", "motor_name",
                             "anomaly_score", "risk_level",
                             "temp_cur", "vib_z_cur"] if c in df_a.columns]
        return {
            "n_anomalies": len(df_a),
            "anomalies":   df_a[cols].dropna(how="all").head(limit).to_dict(orient="records"),
        }

    # ── Historique alertes externes ────────────────────────────────────────
    @app.get("/v1/alerts", tags=["Alertes"],
             summary="Historique des alertes externes envoyées")
    def get_alerts_history(limit: int = 50):
        """
        Retourne les dernières alertes envoyées via email/webhook/SMS.
        Inclut le statut de livraison par canal.
        Nécessite alert_config.json configuré.
        """
        if not ALERTS_ENABLED or _alert_manager is None:
            return {
                "enabled": False,
                "message": "AlertManager non disponible. Vérifier alert_config.json",
                "alerts": []
            }
        return {
            "enabled": True,
            "stats":   _alert_manager.get_stats(),
            "alerts":  _alert_manager.get_history(limit=limit)
        }

    @app.get("/v1/alerts/stats", tags=["Alertes"],
             summary="Statistiques du gestionnaire d'alertes")
    def get_alerts_stats():
        """Statistiques globales : total envoyées, par niveau, cooldowns actifs."""
        if not ALERTS_ENABLED or _alert_manager is None:
            return {"enabled": False, "channels": "aucun", "total_alerts": 0}
        return {"enabled": True, **_alert_manager.get_stats()}

    # ── Limites et lacunes documentées du système ─────────────────────────
    @app.get("/v1/system-limits", tags=["Système"],
             summary="Limites connues et lacunes techniques du système")
    def get_system_limits():
        """
        Documente honnêtement les limitations techniques identifiées.
        Utile pour la transparence et la soutenance PFE.
        """
        return {
            "version": API_VERSION,
            "limits": [
                {
                    "id": "L1",
                    "titre": "Alertes non persistantes",
                    "description": (
                        "Les alertes sont affichées dans le dashboard mais ne sont pas envoyées "
                        "en dehors de l'interface web si alert_config.json n'est pas configuré. "
                        "Le module alert_manager.py supporte email SMTP, webhook Slack/Teams "
                        "et SMS Twilio, mais nécessite une configuration manuelle."
                    ),
                    "statut": "MODULE_DISPONIBLE",
                    "fichier": "alert_manager.py",
                    "activation": "Modifier alert_config.json et redémarrer l'API"
                },
                {
                    "id": "L2",
                    "titre": "4 features d'accélération toujours nulles",
                    "description": (
                        "acc_p2p, acc_z2p, acc_crest, acc_rms = 0.0 dans toutes les prédictions. "
                        "La gateway IFM AL1352 transmet ces valeurs sur un seul axe Y dans une "
                        "sous-clé séparée du JSON, et la consolidation multi-lignes ne les aligne "
                        "pas avec la mesure principale. Ces 4 features ont variance nulle et "
                        "n'apportent aucune information discriminante au modèle."
                    ),
                    "statut": "LIMITATION_MATERIELLE",
                    "impact": "4/25 features neutralisées — PCA les absorbe sans effet"
                },
                {
                    "id": "L3",
                    "titre": "Courant électrique absent",
                    "description": (
                        "current_mean = 0.0 dans 100%% des prédictions. Les capteurs IFM VVB001/VSE002 "
                        "mesurent uniquement vibrations et température. Aucun capteur de courant "
                        "(pince ampèremétrique, transducteur de courant) n'est intégré au banc d'essai. "
                        "Le poids courant (15%%) du health_score est systématiquement neutralisé."
                    ),
                    "statut": "LIMITATION_MATERIELLE",
                    "impact": "health_score calculé sur 85%% de son potentiel — poids courant inactif"
                },
                {
                    "id": "L4",
                    "titre": "Pas de déploiement Edge computing",
                    "description": (
                        "L'API tourne sur un PC Windows standard (localhost:8000). "
                        "Il n'existe pas de Dockerfile, de configuration ARM (Raspberry Pi / Jetson), "
                        "ni de modèles ONNX/TFLite optimisés pour embarqué. "
                        "L'export ONNX des modèles pyod (ECOD) n'est pas nativement supporté "
                        "par sklearn-onnx et nécessiterait une refonte du pipeline d'inférence."
                    ),
                    "statut": "NON_IMPLEMENTE",
                    "solution_envisagee": (
                        "Exporter IF et OCSVM via sklearn-onnx, "
                        "déployer sur Raspberry Pi 4 avec onnxruntime, "
                        "conteneuriser avec Docker."
                    )
                },
                {
                    "id": "L5",
                    "titre": "Faux niveaux URGENT sur capteurs sains",
                    "description": (
                        "La formule RUL était trop sensible : des capteurs avec health_score > 90 "
                        "recevaient un niveau URGENT. Correction appliquée dans cette version : "
                        "tout capteur avec health_score >= 85 est forcé en niveau OK, "
                        "avec un RUL plancher de 500 heures."
                    ),
                    "statut": "CORRIGE_PARTIELLEMENT",
                    "correctif": "Filtre health_score >= 85 → forcer alert_level = OK (compute_rul)"
                },
                {
                    "id": "L6",
                    "titre": "RUL heuristique — pas de modèle entraîné",
                    "description": (
                        "Le RUL est calculé par formule empirique (deg_instant * 0.5 + deg_rate * 0.3 "
                        "+ hist_factor * 0.2), pas par un modèle de régression entraîné. "
                        "Un modèle supervisé (Weibull, LSTM de dégradation, Cox) nécessite "
                        "des données de défaillances réelles confirmées. Aucun des 20 capteurs "
                        "n'a atteint la défaillance complète pendant la période de collecte "
                        "(nov. 2025 → mai 2026), rendant l'entraînement supervisé impossible."
                    ),
                    "statut": "LIMITATION_DONNEES",
                    "impact": "RUL estimatif uniquement — précision non validée sur défaillances réelles"
                }
            ]
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, io
    # Force UTF-8 pour eviter UnicodeEncodeError sur Windows (cp1252)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print("\n" + "=" * 80)
    print("  MAINTENANCE PREDICTIVE -- API UNIFIEE V3.1")
    print("=" * 80)
    print(f"  URL     : http://localhost:8000")
    print(f"  Docs    : http://localhost:8000/docs")
    print(f"  Redoc   : http://localhost:8000/redoc")
    print(f"\n  Endpoints IA :")
    print(f"    POST /v1/predict              -> Anomalie (IF+LOF+OCSVM+ECOD) | vote 2/4")
    print(f"    POST /v1/predict-rul          -> RUL (Remaining Useful Life)")
    print(f"    POST /v1/iot-predict          -> Predict+RUL direct IoT sans BDD [NEW]")
    print(f"    GET  /v1/health-score/{{sensor_id}}  -> Score sante 0-100")
    print(f"    GET  /v1/history/{{sensor_id}}       -> Historique predictions")
    print(f"    GET  /v1/alert-level/{{sensor_id}}   -> Niveau alerte dashboard")
    print(f"\n  Endpoints systeme :")
    print(f"    GET  /health    -> Health check + modeles charges")
    print(f"    GET  /metrics   -> F1=0.4008 | AUC=0.6830 | Acc=0.9232")
    print(f"    GET  /sensors   -> 20 capteurs IFM")
    print(f"    GET  /anomalies -> Anomalies filtrees")
    print("=" * 80 + "\n")

    if not FASTAPI_OK:
        print("Installe les dépendances : pip install fastapi uvicorn pydantic scipy")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
