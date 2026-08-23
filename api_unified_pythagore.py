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
║    GET  /metrics                 → Métriques modèle V3 (AUC=0.9475 CV)          ║
║    GET  /sensors                 → Liste capteurs depuis full_data           ║
║    GET  /anomalies               → Anomalies filtrées                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
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

# ── Chargement .env (ignoré silencieusement si absent — Docker injecte les vars) ─
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Authentification API Key ───────────────────────────────────────────────────
from auth import require_api_key, require_admin_key
from rate_limiter import make_rate_limiter

# ── Authentification utilisateurs (register/login humain, JWT) ────────────────
try:
    from user_auth import router as user_auth_router, init_users_table
    USER_AUTH_OK = True
except Exception as _ua_e:
    USER_AUTH_OK = False
    logging.getLogger(__name__).warning(
        f"user_auth non disponible : {_ua_e} — /v1/auth/* sera absent"
    )

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
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, Request
    from fastapi.responses import JSONResponse
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import HTMLResponse, FileResponse
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
MODEL_HBOS      = MODEL_DIR / "model_hbos_v3.pkl"
MODEL_COPOD     = MODEL_DIR / "model_copod_v3.pkl"
MODEL_META_LR   = MODEL_DIR / "model_meta_lr.pkl"
SCALER_PATH     = MODEL_DIR / "scaler_v3.pkl"
PCA_PATH        = MODEL_DIR / "pca_v3.pkl"
FEATURES_PATH   = MODEL_DIR / "features_v3.pkl"
THRESHOLD_PATH  = MODEL_DIR / "threshold_v3.pkl"
METRICS_PATH    = MODEL_DIR / "metrics_v3.csv"
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
meta_lr       = None  # stacking LogisticRegression combinant les scores des modèles (remplace la moyenne fixe)

# Historique glissant des scores d'anomalie par sensor_id (pour RUL)
# Format : {sensor_id: deque([(timestamp, anomaly_score, confidence), ...])}
anomaly_history: dict = {}
HISTORY_WINDOW = 50   # Nombre de mesures gardées en mémoire par moteur

# Buffer de persistance temporelle par capteur (V9)
# Principe : une anomalie n'est confirmée que si k fenêtres consécutives dépassent le seuil
# → réduit les faux positifs isolés, améliore la précision sans sacrifier le recall
_persistence_buffers: dict = {}   # {sensor_id: deque(maxlen=k)} contient soft_scores récents

# Baseline par capteur — pour normaliser le health score relatif (semaine 2)
# Calculé sur les 20 premières mesures de chaque capteur
sensor_baseline: dict = {}
BASELINE_SAMPLES = 20

# Persistence de l'historique sur disque (survit au redémarrage)
HISTORY_PATH = Path("anomaly_history_persist.json")
PERSIST_INTERVAL = 60   # sauvegarder toutes les 60s -- fenetre de perte de donnees
                         # reduite de 5min a 1min en cas d'arret non propre (crash,
                         # kill -9). Cout : ecriture disque d'un petit JSON (quelques
                         # Ko a dizaines de Ko pour 20 capteurs x 50 entrees max) une
                         # fois par minute -- negligeable.
_last_persist: float = 0.0

# Fenêtres glissantes serveur pour /v1/iot-predict (IoT sans accès base de données)
# {sensor_id: deque([MeasurePoint-like dict, ...], maxlen=IOT_WINDOW_SIZE)}
# IMPORTANT : doit rester alignée avec WINDOW_SIZE de train_model_v3_unsupervised.py
# (=20). Une fenêtre serveur plus courte biaise std/trend/kurtosis/entropie par
# rapport à ce que les modèles ont appris (dérive de distribution silencieuse).
iot_windows: dict = {}
IOT_WINDOW_SIZE = 20

# Buffer brut de vibration_z par capteur, alimenté à chaque /v1/predict — sert
# uniquement à l'analyse spectrale publique en lecture (GET /v1/spectral/{id}),
# qui ne peut pas s'appuyer sur iot_windows (vide : le moteur de production
# réel appelle /v1/predict + /v1/predict-rul, jamais /v1/iot-predict).
_raw_vib_buffers: dict = {}
RAW_VIB_BUFFER_SIZE = 64

# ── Purge des capteurs inactifs (anti fuite memoire) ───────────────────────────
# anomaly_history / sensor_baseline / _persistence_buffers / iot_windows sont
# indexes par sensor_id fourni librement par le client (ex: /v1/iot-predict).
# Sans purge, un appelant qui varie le sensor_id a chaque appel fait grossir
# ces dicts indefiniment -- meme classe de fuite que celle deja corrigee dans
# rate_limiter.py, ici cote etat applicatif plutot que rate limiting.
_sensor_last_seen: dict = {}
_SENSOR_SWEEP_INTERVAL = 300.0   # secondes entre deux balayages
_SENSOR_STALE_AFTER    = 7200.0  # capteur inactif depuis 2h -> purge (sauf capteurs IFM connus)
_last_sensor_sweep: float = 0.0


def _touch_sensor_and_sweep(sensor_id: str) -> None:
    """Marque sensor_id comme actif et purge periodiquement les capteurs
    inactifs et inconnus (jamais les capteurs IFM reels de IFM_KNOWN_IDS)."""
    global _last_sensor_sweep
    import time as _time
    now = _time.time()
    _sensor_last_seen[sensor_id] = now
    if now - _last_sensor_sweep < _SENSOR_SWEEP_INTERVAL:
        return
    _last_sensor_sweep = now
    stale = [
        sid for sid, ts in _sensor_last_seen.items()
        if sid not in IFM_KNOWN_IDS and now - ts > _SENSOR_STALE_AFTER
    ]
    for sid in stale:
        _sensor_last_seen.pop(sid, None)
        anomaly_history.pop(sid, None)
        sensor_baseline.pop(sid, None)
        _persistence_buffers.pop(sid, None)
        iot_windows.pop(sid, None)
        _raw_vib_buffers.pop(sid, None)
        globals().get("_latest_results", {}).pop(sid, None)
    if stale:
        log.info(f"Purge capteurs inactifs (>2h, non-IFM) : {len(stale)} capteur(s)")


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
    "bc59bf5f","d9508e77","eb084747","f48c25f9","ed6fa322",
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
    global models, scaler, pca, features_list, df_results, thresholds, meta_lr

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
        if MODEL_HBOS.exists():
            models["hbos"]  = joblib.load(MODEL_HBOS)
            log.info("✅ HBOS chargé")
        if MODEL_COPOD.exists():
            models["copod"] = joblib.load(MODEL_COPOD)
            log.info("✅ COPOD chargé")
        if THRESHOLD_PATH.exists():
            thresholds = joblib.load(THRESHOLD_PATH)
            log.info(f"✅ Seuils optimaux chargés | input={thresholds.get('unsupervised_input','pca')} | ensemble={thresholds.get('ensemble_names',['IF','LOF','OCSVM','ECOD'])}")
        if MODEL_META_LR.exists():
            meta_lr = joblib.load(MODEL_META_LR)
            log.info("✅ Stacking LogisticRegression chargé (remplace la moyenne fixe SoftVote)")
        n_models = sum(1 for k in ["if","lof","ocsvm","ecod","hbos","copod"] if k in models)
        log.info(f"✅ {n_models} modèles chargés | Features: {len(features_list)} | PCA: {pca.n_components_}")

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
    individual_scores: dict = {}
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
    individual_scores: dict = {}
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
    """Kurtosis "régulière" (fisher=False, baseline=3 pour une distribution
    normale) — DOIT rester alignée avec train_model_v3_unsupervised.py qui
    utilise la même convention (kurtosis(VZ, fisher=False)) pour calculer
    vib_z_kurt/vib_x_kurt/vib_y_kurt. Une divergence de convention ici décale
    silencieusement ces 3 features de ~3 unités par rapport à ce que le
    RobustScaler/PCA/modèles ont appris comme "normal" en entraînement."""
    from scipy.stats import kurtosis as sp_kurt
    if len(lst) < 4:
        return 3.0
    arr = np.array(lst, dtype=float)
    if np.std(arr) < 1e-10:   # données constantes → kurtosis indéfini
        return 3.0
    return float(sp_kurt(arr, fisher=False))

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
        # Courant moteur — np.nan est "truthy" en Python : `NaN or 0.0` reste NaN,
        # donc le fallback à 0.0 (capteurs sans mesure de courant) ne s'appliquait
        # jamais avec `or`. np.nan_to_num() corrige ça côté vecteur ML (nan_to_num
        # dans /v1/predict), mais la réponse JSON "features" renvoyait "null" au
        # lieu de 0.0 — corrigé ici pour que la valeur affichée soit cohérente.
        "current_mean": float(np.nan_to_num(safe_mean([h.current for h in history if h.current is not None]), nan=0.0)),
        "current_std":  float(np.nan_to_num(safe_std([h.current  for h in history if h.current is not None]), nan=0.0)),
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
    # np.nan_to_num plutôt que `or 0.0` : NaN est "truthy" en Python, le
    # fallback `or` ne s'active jamais pour une moyenne indéfinie (axe absent).
    vx_m = float(np.nan_to_num(safe_mean(vib_x), nan=0.0))
    vy_m = float(np.nan_to_num(safe_mean(vib_y), nan=0.0))
    vz_m = float(np.nan_to_num(safe_mean(vib_z), nan=0.0))
    feat["vib_asym_xy"] = float(abs(vx_m - vy_m) / (vx_m + vy_m + 1e-9))
    feat["vib_asym_xz"] = float(abs(vx_m - vz_m) / (vx_m + vz_m + 1e-9))

    return feat


def run_ensemble(X_raw: np.ndarray, sensor_id: str = "default") -> dict:
    """
    Applique scaler → X_scaled → SoftVote continu (IF+ECOD+HBOS+COPOD) + filtre de persistance.
    Score continu [0,1] calculé par score_samples/decision_function + normalisation p1/p99.
    Seuil optimal chargé depuis threshold_v3.pkl (softvote_threshold).
    Persistance k=3 : anomalie confirmée uniquement si k fenêtres consécutives > seuil.
    """
    if not models:
        return {"votes": 0, "label": "NORMAL", "confidence": 0.0,
                "is_anomaly": False, "raw_anomaly": False,
                "individual": {"IF": "N/A", "LOF": "N/A",
                               "OCSVM": "N/A", "ECOD": "N/A",
                               "HBOS": "N/A", "COPOD": "N/A"},
                "individual_scores": {"IF": 0.0, "LOF": 0.0, "OCSVM": 0.0,
                                      "ECOD": 0.0, "HBOS": 0.0, "COPOD": 0.0}}

    X_scaled = scaler.transform(X_raw)
    use_scaled = thresholds.get("unsupervised_input", "pca") == "scaled" if thresholds else False
    X_unsup = X_scaled if use_scaled else pca.transform(X_scaled)

    score_stats = (thresholds or {}).get("score_stats", {})

    def _soft(model, key, is_pyod=False):
        """Score continu normalisé [0,1] : 1 = certain anomalie."""
        try:
            if is_pyod:
                raw = float(model.decision_function(X_unsup)[0])
            else:
                raw = float(-model.score_samples(X_unsup)[0])
            stats = score_stats.get(key)
            if stats and stats["p99"] > stats["p1"]:
                return float(np.clip((raw - stats["p1"]) / (stats["p99"] - stats["p1"]), 0.0, 1.0))
            # Fallback sigmoid si stats absentes (avant premier re-entraînement)
            return float(1.0 / (1.0 + np.exp(-raw)))
        except Exception:
            try:
                pred = model.predict(X_unsup)[0]
                return 1.0 if (pred == -1 or pred == 1) else 0.0
            except Exception:
                return 0.5

    s_if    = _soft(models["if"],    "if",    is_pyod=False)
    s_lof   = _soft(models["lof"],   "lof",   is_pyod=True)
    s_ocsvm = _soft(models["ocsvm"], "ocsvm", is_pyod=False)
    s_ecod  = _soft(models["ecod"],  "ecod",  is_pyod=True)
    s_hbos  = _soft(models["hbos"],  "hbos",  is_pyod=True) if "hbos"  in models else 0.5
    s_copod = _soft(models["copod"], "copod", is_pyod=True) if "copod" in models else 0.5

    # RAPPORT : stacking LogisticRegression (remplace la moyenne fixe à 4 modèles) —
    # apprend le poids de chaque détecteur, y compris LOF/OCSVM (poids appris
    # quasi nul/négatif s'ils sont bruités, plutôt qu'une exclusion manuelle).
    all_scores = {"if": s_if, "lof": s_lof, "ocsvm": s_ocsvm, "ecod": s_ecod,
                  "hbos": s_hbos, "copod": s_copod}
    meta_order = (thresholds or {}).get("meta_feature_order")
    if meta_lr is not None and meta_order:
        feat_vec = np.array([[all_scores[k] for k in meta_order]])
        soft_score = float(meta_lr.predict_proba(feat_vec)[0, 1])
    else:
        # Fallback (modèles pas encore ré-entraînés avec le stacking) : ancienne
        # moyenne IF+ECOD+HBOS+COPOD, LOF/OCSVM exclus (AUC individuel trop faible).
        active_scores = [s_if, s_ecod]
        if "hbos"  in models: active_scores.append(s_hbos)
        if "copod" in models: active_scores.append(s_copod)
        soft_score = float(np.mean(active_scores))

    # Garde-fou NaN/Inf : sous charge concurrente sur un même capteur, un modèle
    # (score_samples/decision_function) peut occasionnellement renvoyer une valeur
    # non finie. Sans ça, ce NaN se propage jusqu'à confidence/anomaly_score dans
    # la réponse -- et Starlette (allow_nan=False) plante en 500 à la sérialisation
    # JSON plutôt que de renvoyer une réponse (observé : ValueError "Out of range
    # float values are not JSON compliant" sous test de charge concurrent).
    if not np.isfinite(soft_score):
        soft_score = 0.5

    # Seuil optimal depuis threshold_v3.pkl, fallback 0.5
    opt_thr    = float((thresholds or {}).get("softvote_threshold", 0.5))
    raw_anomaly = soft_score >= opt_thr   # détection brute (sans filtre)

    # ── V9 : Filtre de persistance temporelle par capteur ────────────────────
    # Un faux positif isolé disparaît après 1 fenêtre ; un vrai défaut persiste k fenêtres.
    k_persist = int((thresholds or {}).get("persistence_k", 3))
    if sensor_id not in _persistence_buffers:
        _persistence_buffers[sensor_id] = deque(maxlen=k_persist)
    _persistence_buffers[sensor_id].append(1 if raw_anomaly else 0)
    buf = _persistence_buffers[sensor_id]
    is_anomaly = (len(buf) >= k_persist and sum(buf) >= k_persist)

    # Votes binaires conservés pour l'affichage individuel par modèle
    def _bin(s): return s >= 0.5
    individual = {
        "IF":    "ANOMALY" if _bin(s_if)    else "NORMAL",
        "LOF":   "ANOMALY" if _bin(s_lof)   else "NORMAL",
        "OCSVM": "ANOMALY" if _bin(s_ocsvm) else "NORMAL",
        "ECOD":  "ANOMALY" if _bin(s_ecod)  else "NORMAL",
        "HBOS":  "ANOMALY" if _bin(s_hbos)  else "NORMAL",
        "COPOD": "ANOMALY" if _bin(s_copod) else "NORMAL",
    }
    votes = sum(1 for v in individual.values() if v == "ANOMALY")
    individual_scores = {
        "IF": round(s_if, 4), "LOF": round(s_lof, 4), "OCSVM": round(s_ocsvm, 4),
        "ECOD": round(s_ecod, 4), "HBOS": round(s_hbos, 4), "COPOD": round(s_copod, 4),
    }
    return {
        "votes":        votes,
        "soft_score":   round(soft_score, 4),
        "label":        "ANOMALY" if is_anomaly else "NORMAL",
        "confidence":   round(soft_score, 4),
        "is_anomaly":   is_anomaly,
        "raw_anomaly":  raw_anomaly,   # détection sans filtre (pour debug)
        "individual":   individual,
        "individual_scores": individual_scores,
    }


def update_history(sensor_id: str, score: float, confidence: float):
    """Enregistre la prédiction dans l'historique glissant du moteur."""
    import math
    _touch_sensor_and_sweep(sensor_id)
    if math.isnan(score) or math.isinf(score):
        score = 0.0
    if math.isnan(confidence) or math.isinf(confidence):
        confidence = 0.0
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
    # +3 par rapport aux seuils historiques (4/7/10) : safe_kurtosis() utilise
    # désormais fisher=False comme le training (baseline ≈3 au lieu de ≈0),
    # donc les seuils de dégradation doivent être décalés d'autant.
    "vib_z_kurt":  {"warn": 7.0,            "crit": 10.0,           "max": 13.0},
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

    # On utilise le risk_level de la prédiction comme référence principale.
    # Chaque branche fixe alert_level ET la plage [deg_lo, deg_hi] de
    # deg_combined qui a réellement produit ce niveau -- cette plage sert
    # ensuite à l'interpolation du RUL (étape 6). C'est nécessaire car les
    # branches ci-dessous n'utilisent PAS toutes les mêmes seuils que
    # RUL_TABLE (ex: ÉLEVÉ/MODÉRÉ assigne "ATTENTION" pour tout deg_combined
    # < 0.55, alors que RUL_TABLE/le fallback l'assigne pour [0.30, 0.55)
    # seulement). Utiliser un ratio fixe basé sur [0.30, 0.55] pour TOUTES
    # les branches -- comme avant -- clampait le ratio à 0 dès que
    # deg_combined < 0.30, ce qui arrive pour la quasi-totalité des capteurs
    # réels passés par les branches CRITIQUE/ÉLEVÉ/MODÉRÉ : sur le parc réel
    # (mariadb_realtime), ça figeait rul_hours=336h (plafond ATTENTION) pour
    # 665/815 mesures (81%), de health_score 42.8 à 84.8 confondus -- deux
    # capteurs à des états de santé opposés recevaient exactement le même
    # RUL affiché. Utiliser la plage réelle de la branche corrige ça.
    predict_risk = (predict_result or {}).get("risk_level", "")
    if predict_risk == "CRITIQUE":
        # CRITIQUE ML : alerte selon deg_combined pour nuancer. Corrigé --
        # avant, cette branche forçait TOUJOURS au moins "URGENT" dès que le
        # ML disait CRITIQUE, même avec une dégradation physique faible
        # (ex: 15%, quasi sain) -- contrairement à la branche ÉLEVÉ/MODÉRÉ qui
        # a 2 paliers de nuance. Le score du modèle sature souvent vers
        # 0.99-1.0 (stacking peu calibré), donc predict_risk="CRITIQUE"
        # arrivait beaucoup trop souvent : ça faisait passer quasi tout le
        # parc en "URGENT" simultanément sur le dashboard (rul_hours=168h
        # identique partout) et déclenchait une alerte email pour presque
        # chaque capteur. Ajout d'un palier ATTENTION en dessous de 0.30 pour
        # refléter une vraie dégradation faible malgré le signal ML.
        if deg_combined >= 0.60:
            alert_level = "CRITIQUE"
            deg_lo, deg_hi = 0.60, 1.0
        elif deg_combined >= 0.30:
            alert_level = "URGENT"
            deg_lo, deg_hi = 0.30, 0.60
        else:
            alert_level = "ATTENTION"
            deg_lo, deg_hi = 0.0, 0.30
    elif predict_risk in ("ÉLEVÉ", "MODÉRÉ"):
        if deg_combined >= 0.55:
            alert_level = "URGENT"
            deg_lo, deg_hi = 0.55, 1.0
        else:
            alert_level = "ATTENTION"
            deg_lo, deg_hi = 0.0, 0.55
    elif predict_risk == "FAIBLE":
        # Même si ML dit FAIBLE, surveiller si dégradation physique élevée
        if deg_combined >= 0.80:
            alert_level = "URGENT"
            deg_lo, deg_hi = 0.80, 1.0
        elif deg_combined >= 0.30:
            alert_level = "ATTENTION"
            deg_lo, deg_hi = 0.30, 0.80
        else:
            alert_level = "OK"
            deg_lo, deg_hi = 0.0, 0.30
    else:
        # Fallback sur deg_combined seul
        if deg_combined >= 0.80:
            alert_level = "CRITIQUE"
            deg_lo, deg_hi = 0.80, 1.0
        elif deg_combined >= 0.55:
            alert_level = "URGENT"
            deg_lo, deg_hi = 0.55, 0.80
        elif deg_combined >= 0.30:
            alert_level = "ATTENTION"
            deg_lo, deg_hi = 0.30, 0.55
        else:
            alert_level = "OK"
            deg_lo, deg_hi = 0.0, 0.30

    # ── 6. Estimation RUL en heures ───────────────────────────────────────
    rul_min, rul_max = RUL_TABLE[alert_level]
    # Interpolation linéaire de deg_combined dans [deg_lo, deg_hi] (la plage
    # réelle qui a produit alert_level, cf. étape 5) vers [rul_min, rul_max].
    # Ratio toujours borné [0,1] pour ne jamais sortir de la plage RUL_TABLE
    # de l'alerte affichée (incohérence sinon avec le texte de recommandation,
    # ex: "RUL 3-7 jours" avec un rul_hours de 307h).
    ratio = min(1.0, max(0.0, (deg_combined - deg_lo) / max(1e-9, deg_hi - deg_lo)))
    rul_hours = rul_max - ratio * (rul_max - rul_min)

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
        if USER_AUTH_OK:
            try:
                init_users_table()
            except Exception as e:
                log.warning(f"init_users_table() a échoué : {e}")
        yield
        # Shutdown : sauvegarder une dernière fois avant arrêt
        import time as _t
        global _last_persist
        _last_persist = 0.0
        save_history_to_disk()
        log.info("API arrêtée proprement — historique sauvegardé")

    class UTF8JSONResponse(JSONResponse):
        """Force le charset=utf-8 dans Content-Type — sans ça, PowerShell 5.1
        (Invoke-RestMethod) décode le JSON en ISO-8859-1 par défaut et corrompt
        les caractères accentués (ex: "ÉLEVÉ" -> "ÃLEVÃ")."""
        media_type = "application/json; charset=utf-8"

    app = FastAPI(
        title="Maintenance Prédictive — API Unifiée",
        description=(
            "Système complet de surveillance de 20 capteurs IFM — Novation City.\n\n"
            "**Modèle IA** : Ensemble non supervisé IF+ECOD+HBOS+COPOD (SoftVote), LOF/OCSVM calculés à titre diagnostic uniquement\n\n"
            "**Données** : Capteurs IFM VVB001 → MySQL ai_cp (1 648 886 mesures, nov 2025 – mar 2026)\n\n"
            "**PFE ISG Bizerte** — Détection d'anomalies + Estimation RUL roulements\n\n"
            "**Auth** : En-tête `X-API-Key` obligatoire sur tous les endpoints /v1/*"
        ),
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        default_response_class=UTF8JSONResponse,
    )

    # CORS restreint aux origines définies via CORS_ORIGINS (prod) ou * (dev)
    _cors_origins_env = os.getenv("CORS_ORIGINS", "")
    _cors_origins = (
        [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
        if _cors_origins_env
        else ["*"]
    )
    if "*" in _cors_origins:
        log.warning("CORS ouvert à toutes les origines — acceptable en dev, à restreindre en prod via CORS_ORIGINS")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Comptes utilisateurs (register/login humain) — additif, n'affecte pas
    # l'auth par clé API (require_api_key/require_admin_key) utilisée ailleurs
    if USER_AUTH_OK:
        app.include_router(user_auth_router, prefix="/v1/auth", tags=["Authentification utilisateurs"])

    # Cache des derniers résultats par capteur (pour /v1/results)
    _latest_results: dict = {}

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
                "GET  /metrics":                  "Métriques modèle (F1=0.298, AUC=0.9475, CV 3-fold)",
                "GET  /sensors":                  "Liste 20 capteurs IFM",
                "GET  /anomalies":                "Anomalies filtrées par score",
            }
        }

    # ── Health Check ───────────────────────────────────────────────────────
    @app.get("/health", tags=["Système"])
    def health():
        return {
            "status":         "ok",
            "models_loaded":  len(models) >= 4,
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
            "d'anomalie basé sur le stacking (LogisticRegression) des 6 modèles IF/LOF/OCSVM/ECOD/HBOS/COPOD.\n\n"
            "**Format données** : compatible avec le champ `data` de la table `full_data` "
            "(SensorNodeId, Temperature, Vibration RMS X/Y/Z)."
        )
    )
    def predict_anomaly(request: Request, req: PredictRequest, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(60))):
        if not req.history:
            raise HTTPException(status_code=400, detail="history ne peut pas être vide")

        # 1. Extraction features
        feat = extract_features(req.history)

        # 1b. Alimente le buffer brut vibration_z (pour GET /v1/spectral/{id},
        # lecture publique — voir _raw_vib_buffers)
        _vib_pts = [h.vibration_z for h in req.history if h.vibration_z is not None]
        if _vib_pts:
            _buf = _raw_vib_buffers.setdefault(req.sensor_id, deque(maxlen=RAW_VIB_BUFFER_SIZE))
            _buf.extend(_vib_pts)

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

        # 3. Inférence ensemble (avec filtre persistance par capteur)
        sensor_id_pred = req.sensor_id if hasattr(req, "sensor_id") and req.sensor_id else "default"
        result = run_ensemble(X, sensor_id=sensor_id_pred)

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

        # Cohérence prediction/risk : tant que le filtre de persistance (V9) n'a
        # pas confirmé l'anomalie (is_anomaly=False), ne pas afficher CRITIQUE/
        # ÉLEVÉ. Le score brut peut déjà être élevé dès la 1re mesure suspecte,
        # avant confirmation sur k fenêtres consécutives -- sans ce plafond, la
        # réponse pouvait annoncer prediction="NORMAL" avec risk_level="CRITIQUE"
        # en même temps (observé en test réel), contradictoire pour un dashboard.
        if not result["is_anomaly"] and risk in ("CRITIQUE", "ÉLEVÉ"):
            risk = "MODÉRÉ"

        # Cohérence health/risk : capteur sain (health ≥ 85) non confirmé anomalie → FAIBLE
        # Évite MODÉRÉ+Health=99 qui est contradictoire pour l'utilisateur
        _health_val = feat.get("health_score", 0) or 0
        if _health_val >= 85 and not result["is_anomaly"]:
            risk = "FAIBLE"

        # 6. Mise à jour historique moteur (pour RUL)
        update_history(req.sensor_id, anomaly_score, result["confidence"])

        # 6b. Pas d'alerte externe ici : le risque d'anomalie (risk_level) ne dit
        # rien du temps restant avant défaillance. Sur demande, les alertes
        # email/webhook/SMS sont désormais déclenchées uniquement depuis
        # /v1/predict-rul, dont alert_level (URGENT/CRITIQUE) correspond
        # précisément à un RUL <= 7 jours -- voir ce endpoint plus bas.

        # 7. Features utiles à retourner (pour debug / dashboard)
        # NaN -> None (jamais la valeur NaN brute) : Starlette sert allow_nan=False,
        # un float NaN dans la réponse fait planter la sérialisation JSON en 500
        # (observé sous charge concurrente sur un même capteur -- ValueError:
        # "Out of range float values are not JSON compliant").
        feat_summary = {
            k: (round(v, 4) if not np.isnan(v) else None) if isinstance(v, float) else v
            for k, v in feat.items()
        }

        _ts = datetime.now().isoformat()
        _latest_results.setdefault(req.sensor_id, {}).update({
            "sensor_id":    req.sensor_id,
            "motor_id":     req.motor_id,
            "timestamp":    _ts,
            "predict": {
                "prediction":        result["label"],
                "is_anomaly":        result["is_anomaly"],
                "confidence":        result["confidence"],
                "anomaly_score":     anomaly_score,
                "risk_level":        risk,
                "votes":             result["votes"],
                "individual_models": result["individual"],
                "individual_scores": result["individual_scores"],
                "features":          feat_summary,
            }
        })

        return PredictResponse(
            sensor_id         = req.sensor_id,
            motor_id          = req.motor_id,
            timestamp         = _ts,
            prediction        = result["label"],
            is_anomaly        = result["is_anomaly"],
            confidence        = result["confidence"],
            votes             = result["votes"],
            risk_level        = risk,
            anomaly_score     = anomaly_score,
            individual_models = result["individual"],
            individual_scores = result["individual_scores"],
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
    def predict_rul(request: Request, req: RULRequest, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(60))):
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

        # 5. Sélection du résultat final
        # Stratégie : heuristique pour les heures RUL (calibrée sur les plages CDC),
        # ML pour détecter une dégradation plus précoce → peut élever le niveau d'alerte.
        # Le ML seul ne fixe plus les heures car son dataset synthétique n'a pas de
        # défaillances réelles → ses valeurs absolues restent surestimées.
        _LEVEL_ORDER = ["OK", "ATTENTION", "URGENT", "CRITIQUE"]
        _predict_risk = (req.risk_level or "OK").upper()
        _ml_alert     = (ml_rul_result or {}).get("alert_level", "OK")
        _heur_alert   = rul_heuristic["alert_level"]

        # Choisir le niveau d'alerte le plus sévère parmi heuristique et ML
        _ml_idx   = _LEVEL_ORDER.index(_ml_alert)   if _ml_alert   in _LEVEL_ORDER else 0
        _heur_idx = _LEVEL_ORDER.index(_heur_alert) if _heur_alert in _LEVEL_ORDER else 0
        alert_level = _LEVEL_ORDER[max(_ml_idx, _heur_idx)]

        # Heures RUL : toujours issues de l'heuristique (plages CDC garanties)
        # On réinterpolele dans la plage du niveau d'alerte final si celui-ci a été
        # aggravé par le ML (le RUL doit alors être dans la plage du nouveau niveau)
        if alert_level != _heur_alert and alert_level in RUL_TABLE:
            _rmin, _rmax = RUL_TABLE[alert_level]
            _deg = rul_heuristic["degradation_rate"] / 100.0
            _ratio = min(1.0, max(0.0, _deg))
            rul_hours = round(max(0.0, _rmax - _ratio * (_rmax - _rmin)), 1)
        else:
            rul_hours = rul_heuristic["rul_hours"]
        rul_days      = round(rul_hours / 24.0, 2)

        _ml_model_used = (ml_rul_result or {}).get("model_type", "none")
        confidence     = "HAUTE" if len(req.history) >= 10 else ("MOYENNE" if len(req.history) >= 5 else "FAIBLE")
        recommendation = rul_heuristic["recommendation"]
        trend_detail   = rul_heuristic["trend"]
        trend_detail["rul_model"] = f"heuristic_CDC + ML_{_ml_model_used}"

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

        _ts_rul = datetime.now().isoformat()
        _latest_results.setdefault(req.sensor_id, {}).update({
            "sensor_id": req.sensor_id,
            "timestamp": _ts_rul,
            "rul": {
                "rul_hours":        rul_hours,
                "rul_days":         rul_days,
                "health_score":     rul_heuristic["health_score"],
                "degradation_rate": rul_heuristic["degradation_rate"],
                "alert_level":      alert_level,
                "recommendation":   recommendation,
                "confidence":       confidence,
            }
        })

        return RULResponse(
            sensor_id        = req.sensor_id,
            motor_id         = req.motor_id,
            timestamp        = _ts_rul,
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
    def iot_predict(request: Request, req: IoTMeasurementRequest, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(60))):
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

        # 5. Inférence ensemble (avec filtre persistance par capteur)
        result = run_ensemble(X, sensor_id=req.sensor_id)

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

        # Cohérence prediction/risk — voir /v1/predict pour l'explication complète.
        if not result["is_anomaly"] and risk in ("CRITIQUE", "ÉLEVÉ"):
            risk = "MODÉRÉ"

        _health_val = feat.get("health_score", 0) or 0
        if _health_val >= 85 and not result["is_anomaly"]:
            risk = "FAIBLE"

        # 8. Mise à jour historique anomalies
        update_history(req.sensor_id, anomaly_score, result["confidence"])

        # 9. RUL — calculé uniquement si >= 3 mesures disponibles
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

        # 10. Alerte externe — uniquement si RUL <= 7 jours (alert_level
        # URGENT ou CRITIQUE, voir RUL_TABLE). Le risque d'anomalie seul
        # (risk_level) ne dit rien du temps restant avant défaillance, donc
        # ne déclenche plus d'alerte ici (aligné avec /v1/predict-rul).
        if ALERTS_ENABLED and _alert_manager and alert_level in ("URGENT", "CRITIQUE"):
            _alert_manager.send_alert(
                sensor_id    = req.sensor_id,
                risk_level   = alert_level,
                health_score = health_score if health_score is not None else feat.get("health_score", 0),
                rul_hours    = rul_hours,
                vib_total    = feat.get("vib_total"),
                temperature  = feat.get("temp_cur"),
                votes        = result["votes"]
            )

        # 11. Features résumées
        # NaN -> None, jamais la valeur brute (voir le meme correctif sur /v1/predict).
        feat_summary = {
            k: (round(v, 4) if not np.isnan(v) else None) if isinstance(v, float) else v
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
            individual_scores = result["individual_scores"],
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
    def get_health_score(request: Request, sensor_id: str, _rl=Depends(make_rate_limiter(60))):
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
        scores = [e["score"] for e in hist if not np.isnan(e.get("score", np.nan))]
        if not scores:
            return {"sensor_id": sensor_id, "health_score": 100.0, "status": "Scores invalides", "n_records": 0}
        recent = scores[-10:]

        # Score brut
        raw_health = 100 * (1 - float(np.nanmean(recent)))

        # Normalisation par baseline capteur — corrige le biais global 43-48
        # Si baseline connue : on recentre le score autour de 100 (baseline = 0% dégradation)
        baseline = sensor_baseline.get(sensor_id)
        if baseline is not None and baseline > 0 and not np.isnan(baseline):
            # Score relatif : combien on a dégradé par rapport à la baseline
            degradation_relative = max(0.0, float(np.nanmean(recent)) - baseline)
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
    def get_metrics(request: Request, _rl=Depends(make_rate_limiter(30))):
        """
        Retourne les métriques de performance du modèle non supervisé.
        Source : models/metrics_v3.csv (F1=0.298, AUC=0.9475, CV 3-fold par capteur)
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
            # RAPPORT : stacking LogisticRegression sur tous les modèles chargés
            softvote_models = [k for k in ["if","lof","ocsvm","ecod","hbos","copod"] if k in models]
            n_models = len(softvote_models)
            ens_names = m.get("ensemble", " + ".join(k.upper() for k in softvote_models))
            # Coefficients réels du stacking (weights_if, weights_lof, ... dans metrics_v3.csv)
            # — fallback à un poids égal si le modèle n'a pas encore été ré-entraîné avec le stacking.
            learned_weights = {
                k.upper(): round(float(m[f"weights_{k}"]), 4)
                for k in softvote_models if f"weights_{k}" in m
            }
            weights_out = learned_weights or {k.upper(): round(1/max(n_models,1), 2) for k in softvote_models}
            return {
                "model_version": m.get("model_version", "V7"),
                "ensemble":      ens_names,
                "voting":        m.get("voting", "SoftVote seuil optimal"),
                "dataset":       m.get("dataset", "ai_cp full_data — 20 capteurs IFM"),
                "window_size":   int(float(m.get("window_size", 20))),
                "n_features":    int(float(m.get("n_features",  31))),
                "augmentation":  m.get("augmentation", "x3"),
                "pca_variance":  float(m.get("pca_variance", 0.95)),
                "f1_score":      round(float(m.get("f1_score",  0)), 4),
                "accuracy":      round(float(m.get("accuracy",  0)), 4),
                "precision":     round(float(m.get("precision", 0)), 4),
                "recall":        round(float(m.get("recall",    0)), 4),
                "auc_roc":       round(float(m.get("auc_roc",   0)), 4),
                "n_anomalies":   int(float(m.get("n_anomalies", 0))),
                "n_total":       int(float(m.get("n_total",     0))),
                "contamination": round(float(m.get("contamination", 0)), 4),
                "weights": weights_out,
                "source_file": str(path.name),
            }
        except Exception as e:
            return {"message": f"Erreur lecture métriques : {e}"}

    # ── Fiche technique du modèle (model card) — traçabilité pour production ──
    @app.get("/v1/model-card", tags=["Système"],
             summary="Fiche technique : provenance des données, méthodologie, limites connues")
    def get_model_card(request: Request, _rl=Depends(make_rate_limiter(30))):
        """
        Expose de façon structurée et programmatique la provenance des données
        d'entraînement et les limites de chaque modèle -- pratique standard
        (« model card » / « data sheet ») pour tout système ML utilisé en
        production, en particulier quand un modèle (ici le RUL) est entraîné
        sur des données synthétiques faute de vraies données de panne.

        Objectif : qu'un intégrateur tiers puisse vérifier PROGRAMMATIQUEMENT
        (pas seulement dans une doc PDF qu'on peut oublier de lire) sur quoi
        repose une prédiction avant de l'utiliser pour une décision critique.
        """
        card = {
            "generated_at": datetime.now().isoformat(),
            "api_version": API_VERSION,
            "anomaly_detection": {
                "models": ["IsolationForest", "LOF", "OCSVM", "ECOD", "HBOS", "COPOD"],
                "fusion": "Stacking LogisticRegression",
                "data_provenance": {
                    "source": "ai_cp.full_data — mesures réelles capteurs IFM (20 capteurs, Novation City)",
                    "real_measurements": True,
                    "label_type": "heuristique (percentiles composites, PAS de panne confirmée étiquetée)",
                    "label_caveat": (
                        "Les 'anomalies' d'entraînement sont définies par des seuils statistiques "
                        "(percentile de vibration/température/kurtosis), pas par des pannes réelles "
                        "confirmées. Le modèle apprend à reproduire cette règle, pas à prédire des "
                        "défaillances observées."
                    ),
                },
                "evaluation": {},
            },
            "rul_estimation": {
                "components": ["heuristique (seuils industriels fixes)", "GradientBoostingRegressor (ML)"],
                "data_provenance": {
                    "source": "Courbes de dégradation SYNTHÉTIQUES (loi de Weibull), recalibrées sur la distribution de mesures réelles",
                    "real_failure_samples": 0,
                    "real_measurements": False,
                    "caveat": (
                        "AUCUNE donnée de défaillance réelle confirmée n'a été utilisée -- aucun "
                        "moteur n'a atteint la panne complète pendant la période de collecte "
                        "(nov. 2025 - juin 2026). Les heures de RUL retournées sont des estimations "
                        "non validées empiriquement contre de vraies pannes. À ne pas utiliser comme "
                        "seule base d'une décision d'arrêt machine sans jugement d'un technicien qualifié."
                    ),
                },
                "evaluation": {},
            },
            "known_limitations": [
                "current_mean systématiquement à 0 — aucun capteur de courant électrique installé",
                "Détection binaire uniquement (anomalie/normal) — pas de classification du type de défaut (bille/piste intérieure/extérieure)",
                "Rate limiting par IP peu fiable derrière un reverse proxy sans configuration proxy_headers",
            ],
        }

        # Metriques anomalies -- reutilise le meme fichier que /metrics
        try:
            path = METRICS_PATH if Path(METRICS_PATH).exists() else None
            if path:
                df_m = pd.read_csv(path, encoding="latin-1")
                m = df_m.set_index("metric")["value"].to_dict() if "metric" in df_m.columns else df_m.iloc[0].to_dict()
                card["anomaly_detection"]["evaluation"] = {
                    "auc_roc":    round(float(m.get("auc_roc", 0)), 4),
                    "f1_score":   round(float(m.get("f1_score", 0)), 4),
                    "precision":  round(float(m.get("precision", 0)), 4),
                    "recall":     round(float(m.get("recall", 0)), 4),
                    "n_total":    int(float(m.get("n_total", 0))),
                    "n_anomalies": int(float(m.get("n_anomalies", 0))),
                    "trained_at": m.get("trained_at", "inconnu"),
                    "evaluation_method": m.get("evaluation", "inconnu"),
                }
        except Exception as e:
            card["anomaly_detection"]["evaluation"] = {"error": str(e)}

        # Metriques RUL
        try:
            rul_metrics_path = MODEL_DIR / "metrics_rul_v1.json"
            if rul_metrics_path.exists():
                rm = json.loads(rul_metrics_path.read_text(encoding="utf-8"))
                card["rul_estimation"]["evaluation"] = {
                    "r2_test":     rm.get("r2_test"),
                    "mae_test_h":  rm.get("mae_test_h"),
                    "mae_test_pct": rm.get("mae_test_pct"),
                    "rmse_test_h": rm.get("rmse_test_h"),
                    "n_train":     rm.get("n_train"),
                    "n_test":      rm.get("n_test"),
                    "trained_at":  rm.get("trained_at", "inconnu"),
                }
        except Exception as e:
            card["rul_estimation"]["evaluation"] = {"error": str(e)}

        return card

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
    def spectral_analysis(request: Request, req: PredictRequest, rpm: float = 1450.0, fs: float = 100.0, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(20))):
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
    #  GET /v1/spectral/{sensor_id} — Analyse spectrale publique (dashboard)
    # ══════════════════════════════════════════════════════════════════════
    @app.get(
        "/v1/spectral/{sensor_id}",
        tags=["IA / Prédiction"],
        summary="Analyse spectrale publique à partir du buffer serveur",
        description=(
            "Version publique (lecture seule, sans clé) de l'analyse spectrale — "
            "utilise les dernières valeurs de vibration_z déjà reçues via /v1/predict "
            "pour ce capteur (pas de recalcul côté client). Pensée pour le dashboard."
        )
    )
    def spectral_public(request: Request, sensor_id: str, rpm: float = 1450.0, fs: float = 100.0,
                         _rl=Depends(make_rate_limiter(30))):
        if not SIGNAL_PROCESSING_OK:
            return {"available": False, "reason": "Module signal_processing non disponible."}
        buf = _raw_vib_buffers.get(sensor_id)
        if not buf or len(buf) < 8:
            return {"available": False, "reason": f"Pas assez de mesures en buffer ({len(buf) if buf else 0}/8 min)."}
        try:
            result = full_signal_pipeline(vib_signal=list(buf), fs=fs, rpm=rpm, include_raw_spectra=True)
            return {
                "available":         True,
                "sensor_id":         sensor_id,
                "timestamp":         datetime.now().isoformat(),
                "signal_length":     len(buf),
                "analysis_params":   {"fs_hz": fs, "rpm": rpm},
                "spectral_features": result["spectral_features"],
                "bearing_analysis":  result["bearing_analysis"],
                "raw_spectra":       result.get("raw_spectra", {}),
                "metadata":          result["metadata"],
            }
        except Exception as e:
            log.error(f"Erreur analyse spectrale publique ({sensor_id}) : {e}")
            return {"available": False, "reason": f"Erreur analyse : {str(e)}"}

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
        request: Request,
        type: str = "daily",
        format: str = "json",
        sensor_id: Optional[str] = None,
        _key: str = Depends(require_api_key),
        _rl=Depends(make_rate_limiter(20)),
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

    # ── Résultats JSON temps réel (pour encadrant / export) ───────────────
    @app.get(
        "/v1/results",
        tags=["IA / Prédiction"],
        summary="Derniers résultats de tous les capteurs (GET — navigateur)",
        description=(
            "Retourne en JSON les dernières prédictions (anomalie + RUL) "
            "pour chaque capteur actif. Endpoint GET accessible directement "
            "depuis un navigateur ou curl. Mis à jour à chaque appel /v1/predict."
        )
    )
    def get_results(request: Request, sensor_id: str = None, _rl=Depends(make_rate_limiter(60))):
        """
        GET /v1/results          → tous les capteurs actifs
        GET /v1/results?sensor_id=8f7f2f7e  → un capteur précis
        """
        if not _latest_results:
            return {
                "status":  "en_attente",
                "message": "Aucune prédiction reçue pour l'instant. Démarrez le moteur temps réel.",
                "results": []
            }
        if sensor_id:
            entry = _latest_results.get(sensor_id)
            if not entry:
                raise HTTPException(status_code=404, detail=f"Capteur '{sensor_id}' non trouvé dans les résultats actifs.")
            return jsonable_encoder({
                "status":    "ok",
                "n_sensors": 1,
                "results":   [entry]
            })
        return jsonable_encoder({
            "status":    "ok",
            "timestamp": datetime.now().isoformat(),
            "api":       "Maintenance Prédictive — ISG Bizerte",
            "version":   API_VERSION,
            "n_sensors": len(_latest_results),
            "results":   list(_latest_results.values())
        })

    # ── Liste capteurs ────────────────────────────────────────────────────
    @app.get("/sensors", tags=["Données"])
    def get_sensors(request: Request, _rl=Depends(make_rate_limiter(60))):
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
    def get_history(request: Request, sensor_id: str, limit: int = 20, _rl=Depends(make_rate_limiter(60))):
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

        scores = [e["score"] for e in hist if not np.isnan(e.get("score", np.nan))]
        if not scores:
            scores = [0.0]
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
    def get_alert_level(request: Request, sensor_id: str, _rl=Depends(make_rate_limiter(60))):
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
        scores = [e["score"] for e in hist[-10:] if not np.isnan(e.get("score", np.nan))]
        if not scores:
            return {"sensor_id": sensor_id, "alert_level": "INCONNU", "color": "gray",
                    "message": "Scores invalides.", "timestamp": datetime.now().isoformat()}
        avg    = float(np.nanmean(scores))
        if np.isnan(avg): avg = 0.0
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
    def get_anomalies(request: Request, min_score: float = 0.5, limit: int = 100, _rl=Depends(make_rate_limiter(60))):
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
    def get_alerts_history(request: Request, limit: int = 50, _rl=Depends(make_rate_limiter(30))):
        """
        Retourne les dernières alertes envoyées via email/webhook/SMS.
        Inclut le statut de livraison par canal (booléens uniquement, jamais
        d'adresse/URL/identifiant — public, comme les autres endpoints de
        monitoring en lecture seule).
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
    def get_alerts_stats(request: Request, _rl=Depends(make_rate_limiter(30))):
        """Statistiques globales : total envoyées, par niveau, cooldowns actifs."""
        if not ALERTS_ENABLED or _alert_manager is None:
            return {"enabled": False, "channels": "aucun", "total_alerts": 0}
        return {"enabled": True, **_alert_manager.get_stats()}

    # ── Limites et lacunes documentées du système ─────────────────────────
    @app.get("/v1/system-limits", tags=["Système"],
             summary="Limites connues et lacunes techniques du système")
    def get_system_limits(request: Request, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(30))):
        """
        Documente honnêtement les limitations techniques identifiées.
        Utile pour la transparence et la soutenance PFE.
        """
        return {
            "version": API_VERSION,
            "limits": [
                {
                    "id": "L1",
                    "titre": "Alertes externes — email actif, webhook/SMS non configurés",
                    "description": (
                        "Canal email SMTP configuré et vérifié (envoi réel testé avec succès, "
                        "cooldown 300s/capteur, seuil health_score<=85, niveaux URGENT/CRITIQUE "
                        "uniquement — cohérent avec RUL <= 7 jours). Webhook (Slack/Teams/Discord) "
                        "et SMS (Twilio) restent désactivés faute de compte externe fourni — le "
                        "module alert_manager.py les supporte déjà si besoin plus tard."
                    ),
                    "statut": "EMAIL_ACTIF",
                    "fichier": "alert_manager.py",
                    "activation_restante": "Webhook/SMS : renseigner url/credentials dans alert_config.json",
                    "note_deploiement": "Avant remise au client : remplacer les destinataires email de test par les vraies adresses du client dans alert_config.json"
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
                },
                {
                    "id": "L7",
                    "titre": "BPFO/BPFI/BSF non mesurables avec le fs par défaut de l'analyse spectrale",
                    "description": (
                        "GET /v1/spectral/{id} et POST /v1/spectral-analysis utilisent fs=100Hz par "
                        "défaut (Nyquist = 50Hz). Pour un moteur à 1450 tr/min sur roulement SKF "
                        "6205-2RS, BPFO≈86Hz, BPFI≈131Hz et BSF≈56Hz dépassent tous cette limite : "
                        "leur SNR est structurellement 0 (aucun bin de fréquence n'existe à ces "
                        "valeurs), pas une preuve d'absence de défaut. Seule FTF≈9.6Hz (défaut de "
                        "cage) est mesurable. Cause plus profonde : le signal analysé est la suite "
                        "des vibration_z reçues à chaque appel /v1/predict (espacées de plusieurs "
                        "secondes), pas une forme d'onde d'accéléromètre haute fréquence -- "
                        "augmenter fs ne suffirait pas sans un vrai capteur haute fréquence en amont."
                    ),
                    "statut": "LIMITATION_MATERIELLE",
                    "correctif": "Champ 'measurable' ajouté par fréquence (signal_processing.py) pour distinguer 'non mesurable' de 'absent'",
                    "impact": "Diagnostic BPFO/BPFI/BSF non fiable en l'état — seul FTF est théoriquement exploitable"
                }
            ]
        }


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE UPLOAD — SQL → CSV → Entraînement → Prédictions
# ══════════════════════════════════════════════════════════════════════════════

if FASTAPI_OK:
    import threading, subprocess, tempfile, time as _pipeline_time, uuid as _uuid
    import sys as _sys
    from datetime import datetime as _dt

    # Stockage en mémoire des jobs pipeline (clé = job_id)
    _pipeline_jobs: dict = {}

    def _fmt_elapsed(start_ts: float) -> str:
        s = int(_pipeline_time.time() - start_ts)
        return f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"

    import re as _re
    _ANSI_RE = _re.compile(r'\x1b\[[0-9;]*[mGKHF]|\x1b\(B')

    def _strip_ansi(s: str) -> str:
        return _ANSI_RE.sub("", s)

    def _add_log(job: dict, text: str, log_type: str = ""):
        job["logs"].append({"text": _strip_ansi(text), "type": log_type})

    def _run_pipeline_bg(job_id: str, sql_path: str, train_mode: str):
        """Thread background — exécute toutes les étapes du pipeline."""
        job = _pipeline_jobs[job_id]

        # Chaque run est sauvegardé dans son propre dossier horodaté sous
        # models/runs/ -- ne touche JAMAIS models/ (modèles de production
        # utilisés par les vraies prédictions de l'API, chargés une fois au
        # démarrage). Un upload de test n'écrase donc plus jamais le modèle
        # de prod, et chaque run garde une trace distincte de ses métriques
        # au lieu de se faire silencieusement remplacer par le suivant.
        run_dir = Path("models/runs") / f"{_dt.now():%Y%m%d_%H%M%S}_{job_id}"
        job["run_dir"] = str(run_dir)

        try:
            start = _pipeline_time.time()
            # ── Étape 2 : Parsing SQL → CSV ───────────────────────────────
            job["step"] = 2
            job["step_name"] = "Parsing du fichier SQL..."
            job["progress"]  = 8
            csv_path = sql_path.replace(".sql", "_dataset.csv")

            _add_log(job, "🔍 Démarrage du parsing SQL (extraction mesures IFM)...", "i")

            # LOKY_MAX_CPU_COUNT/OMP_NUM_THREADS : sans ça, train_model_v3_unsupervised.py
            # (lancé en sous-processus juste après) reste bloqué indéfiniment sur cette
            # machine -- sklearn/joblib déclenchent un appel `wmic` pour détecter le
            # nombre de coeurs physiques qui ne rend jamais la main ici (voir incident
            # de blocage résolu manuellement lors de la mise au point de ce pipeline).
            _env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "LOKY_MAX_CPU_COUNT": str(os.cpu_count() or 4),
                "OMP_NUM_THREADS": str(os.cpu_count() or 4),
            }
            proc = subprocess.Popen(
                [_sys.executable, "-u", "generate_dataset_from_sql.py",
                 "--sql", sql_path, "--out", csv_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=_env,
                cwd=str(Path(__file__).parent)
            )
            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                ltype = "s" if "✅" in line else "e" if "❌" in line else "w" if "⚠" in line else "i" if "ℹ" in line else ""
                _add_log(job, line, ltype)
                job["elapsed"] = _fmt_elapsed(start)
                # Avancer la barre pendant le parsing
                job["progress"] = min(28, job["progress"] + 1)

            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("generate_dataset_from_sql.py a échoué (code " + str(proc.returncode) + ")")

            # ── Étape 3 : CSV généré ──────────────────────────────────────
            job["step"] = 3
            job["step_name"] = "Dataset CSV généré"
            job["progress"]  = 30
            csv_size = Path(csv_path).stat().st_size if Path(csv_path).exists() else 0
            _add_log(job, f"✅ CSV généré : {csv_path} ({csv_size/1e6:.1f} MB)", "s")

            # ── Étape 4 : Entraînement des modèles ───────────────────────
            job["step"] = 4
            job["step_name"] = "Entraînement des modèles ML..."
            job["progress"]  = 32
            _add_log(job, "🧠 Lancement de l'entraînement (IF · LOF · OCSVM · ECOD)...", "i")

            # --csv (pas --sql) : le CSV vient d'être correctement généré à
            # l'étape 2/3 par generate_dataset_from_sql.py. Passer --sql ici
            # ferait reparser le fichier SQL brut avec la logique interne de
            # train_model_v3_unsupervised.py (motor_mesure/motor_measurements
            # -- un schéma différent de full_data, qui n'en extrait presque
            # rien : ~400 sessions au lieu des 600 000+ déjà dans le CSV).
            proc2 = subprocess.Popen(
                [_sys.executable, "-u", "train_model_v3_unsupervised.py",
                 "--csv", csv_path, "--out-dir", str(run_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=_env,
                cwd=str(Path(__file__).parent)
            )
            progress_markers = {
                "Isolation Forest": 45, "isolation": 45,
                "LOF": 58, "local outlier": 58,
                "OCSVM": 68, "one-class": 68,
                "ECOD": 76, "HBOS": 80, "COPOD": 83,
                "sauvegarde": 88, "saved": 88, "modèles entraînés": 88,
                "cross-val": 52,
            }
            for raw in proc2.stdout:
                line = raw.strip()
                if not line:
                    continue
                ltype = "s" if "✅" in line else "e" if "❌" in line else "w" if "⚠" in line else "i" if "ℹ" in line else ""
                _add_log(job, line, ltype)
                low = line.lower()
                for marker, pct in progress_markers.items():
                    if marker.lower() in low and job["progress"] < pct:
                        job["progress"] = pct
                        break
                job["elapsed"] = _fmt_elapsed(start)

            proc2.wait()
            if proc2.returncode != 0:
                raise RuntimeError("train_model_v3_unsupervised.py a échoué (code " + str(proc2.returncode) + ")")

            # ── Étape 5 : Confirmation du dossier de sortie ───────────────
            # PAS de load_all_models() ici : ce run reste isolé dans run_dir,
            # les modèles de production (models/) et l'API en cours ne sont
            # pas affectés. Voir la note plus haut sur run_dir.
            job["step"] = 5
            job["step_name"] = "Modèles sauvegardés (hors production)..."
            job["progress"]  = 92
            _add_log(job, f"💾 Modèles et métriques écrits dans {run_dir}/ (production non affectée)", "s")

            # ── Collecte des résultats finaux ─────────────────────────────
            job["step"] = 6
            job["step_name"] = "Pipeline terminé ✅"
            job["progress"]  = 100
            job["elapsed"]   = _fmt_elapsed(start)

            # Métriques du run qui vient de se terminer — lues depuis run_dir/
            # (pas models/metrics_v3.csv, qui reste celui de la production et
            # n'est plus touché par ce pipeline). Ancien code cherchait un
            # "metrics_v3.json" qui n'a jamais existé dans ce projet (seul le
            # .csv existe) : il retombait donc silencieusement sur
            # metrics_rul_v1.json — les métriques d'un AUTRE modèle
            # (régression RUL, pas détection d'anomalies) — et affichait des
            # valeurs sans rapport avec le pipeline qui venait de tourner
            # (toujours les mêmes 6000/600 issus du n_train du modèle RUL,
            # jamais recalculées).
            results: dict = {"run_dir": str(run_dir)}
            metrics_csv = run_dir / "metrics_v3.csv"
            if metrics_csv.exists():
                try:
                    df_m = pd.read_csv(metrics_csv, encoding="latin-1")
                    m = df_m.set_index("metric")["value"].to_dict()
                    results["auc"]          = float(m.get("auc_roc", 0.0))
                    results["f1"]           = float(m.get("f1_score", 0.0))
                    results["n_measures"]   = int(float(m.get("n_total", 0)))
                    results["n_anomalies"]  = int(float(m.get("n_anomalies", 0)))
                except Exception as _metrics_err:
                    _add_log(job, f"⚠️ Lecture metrics_v3.csv échouée : {_metrics_err}", "w")

            if "auc" not in results:
                results.update({"auc": 0.0, "f1": 0.0, "n_measures": 0, "n_anomalies": 0})

            # Top anomalies depuis df_results si disponible
            try:
                if df_results is not None and not df_results.empty:
                    top = df_results.nlargest(8, "anomaly_score")
                    results["top_anomalies"] = [
                        {
                            "sensor_id": str(row.get("sensor_id", "—")),
                            "score": float(row.get("anomaly_score", 0)),
                            "risk":  str(row.get("risk_level", "—")),
                            "temp":  float(row.get("temp_cur", 0)),
                            "vib_z": float(row.get("vib_z_cur", 0)),
                        }
                        for _, row in top.iterrows()
                    ]
            except Exception:
                results["top_anomalies"] = []

            job["results"] = results
            job["status"]  = "done"
            _add_log(job, f"🎉 Pipeline terminé en {job['elapsed']} — AUC={results.get('auc', '?'):.3f}", "s")

        except Exception as exc:
            job["status"] = "error"
            _add_log(job, f"❌ Erreur fatale : {exc}", "e")
            log.error(f"Pipeline {job_id} failed: {exc}")
        finally:
            # Nettoyage fichiers temporaires
            for p in [sql_path, locals().get('csv_path')]:
                try:
                    if p and Path(p).exists():
                        Path(p).unlink()
                except Exception:
                    pass

    # ── Endpoint : page HTML pipeline ─────────────────────────────────────────
    @app.get("/pipeline", tags=["Pipeline"], include_in_schema=False)
    def get_pipeline_page():
        """Sert la page web d'upload SQL."""
        html_path = Path(__file__).parent / "pipeline_upload.html"
        if html_path.exists():
            return FileResponse(str(html_path), media_type="text/html")
        return HTMLResponse("<h1>pipeline_upload.html introuvable</h1>", status_code=404)

    # ── Pages login/register — servies directement par l'API (fonctionnent
    # même sans le conteneur "dashboard" nginx, même pattern que /pipeline).
    # Chemins avec suffixe .html (pas juste /login) pour que les liens relatifs
    # login.html <-> register.html restent valides, qu'ils soient servis par
    # nginx (dashboard, volumes montés sous ce même nom) ou par l'API ici. ──
    @app.get("/login.html", tags=["Authentification utilisateurs"], include_in_schema=False)
    def get_login_page():
        html_path = Path(__file__).parent / "login.html"
        if html_path.exists():
            return FileResponse(str(html_path), media_type="text/html")
        return HTMLResponse("<h1>login.html introuvable</h1>", status_code=404)

    @app.get("/register.html", tags=["Authentification utilisateurs"], include_in_schema=False)
    def get_register_page():
        html_path = Path(__file__).parent / "register.html"
        if html_path.exists():
            return FileResponse(str(html_path), media_type="text/html")
        return HTMLResponse("<h1>register.html introuvable</h1>", status_code=404)

    # ── Endpoint : upload SQL + lancement pipeline ────────────────────────────
    @app.post(
        "/v1/pipeline/upload",
        tags=["Pipeline"],
        summary="Upload SQL + lancement du pipeline complet",
        description=(
            "Reçoit un fichier .sql (dump MariaDB ai_cp), le sauvegarde, "
            "et démarre en arrière-plan :\n"
            "1. Parsing SQL → CSV (generate_dataset_from_sql.py)\n"
            "2. Entraînement des modèles (train_model_v3_unsupervised.py)\n"
            "3. Rechargement des modèles dans l'API\n\n"
            "Retourne un `job_id` à interroger via `GET /v1/pipeline/status/{job_id}`."
        )
    )
    async def pipeline_upload(
        request: Request,
        file: UploadFile = File(..., description="Fichier .sql (dump MariaDB ai_cp)"),
        train_mode: str = Form("full", description="'full' ou 'fast'"),
        _key: str = Depends(require_admin_key),
        _rl=Depends(make_rate_limiter(5)),
    ):
        if not file.filename.lower().endswith(".sql"):
            raise HTTPException(status_code=400, detail="Seuls les fichiers .sql sont acceptés.")

        job_id = str(_uuid.uuid4())[:12]

        # Sauvegarder le fichier SQL uploadé dans /tmp
        tmp_dir = Path(tempfile.gettempdir())
        sql_path = str(tmp_dir / f"pipeline_{job_id}.sql")

        # Écriture en flux par blocs de 4 Mo -- `await file.read()` sans argument
        # chargeait le fichier ENTIER en mémoire (jusqu'à ~650 Mo pour le dump ai_cp)
        # avant de le réécrire sur disque, ce qui ralentissait disproportionnellement
        # les gros uploads (non-linéaire avec la taille) et risquait un OOM sur un
        # déploiement à mémoire limitée (ex: Render free tier).
        CHUNK_SIZE = 4 * 1024 * 1024
        total_bytes = 0
        with open(sql_path, "wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                f.write(chunk)
                total_bytes += len(chunk)

        file_mb = total_bytes / 1e6

        # Créer le job
        _pipeline_jobs[job_id] = {
            "job_id":    job_id,
            "status":    "running",
            "step":      1,
            "step_name": "Fichier reçu",
            "progress":  5,
            "logs":      [{"text": f"📁 Fichier reçu : {file.filename} ({file_mb:.1f} MB)", "type": "s"}],
            "results":   None,
            "elapsed":   "0s",
            "created_at": _dt.now().isoformat(),
            "filename":  file.filename,
        }

        # Lancer le pipeline en background (thread)
        t = threading.Thread(
            target=_run_pipeline_bg,
            args=(job_id, sql_path, train_mode),
            daemon=True
        )
        t.start()

        return {
            "job_id":   job_id,
            "status":   "running",
            "filename": file.filename,
            "size_mb":  round(file_mb, 2),
            "poll_url": f"/v1/pipeline/status/{job_id}",
        }

    # ── Endpoint : status polling ──────────────────────────────────────────────
    @app.get(
        "/v1/pipeline/status/{job_id}",
        tags=["Pipeline"],
        summary="Statut du pipeline en cours",
        description="Interroger toutes les 2-3 secondes. Passer `since=N` pour récupérer uniquement les nouveaux logs (N = index depuis le dernier appel)."
    )
    def pipeline_status(request: Request, job_id: str, since: int = 0, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(60))):
        if job_id not in _pipeline_jobs:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' introuvable.")
        job = _pipeline_jobs[job_id]
        all_logs = job.get("logs", [])
        new_logs = all_logs[since:]
        return {
            "job_id":    job_id,
            "status":    job["status"],
            "step":      job["step"],
            "step_name": job["step_name"],
            "progress":  job["progress"],
            "elapsed":   job["elapsed"],
            "new_logs":  new_logs,
            "total_logs": len(all_logs),
            "results":   job.get("results"),
            "created_at": job.get("created_at"),
        }

    # ── Endpoint : liste des jobs ──────────────────────────────────────────────
    @app.get("/v1/pipeline/jobs", tags=["Pipeline"], summary="Liste des pipelines récents")
    def pipeline_jobs_list(request: Request, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(30))):
        return {
            "total": len(_pipeline_jobs),
            "jobs": [
                {
                    "job_id":   jid,
                    "status":   j["status"],
                    "filename": j.get("filename"),
                    "progress": j["progress"],
                    "elapsed":  j["elapsed"],
                    "created_at": j.get("created_at"),
                }
                for jid, j in list(_pipeline_jobs.items())[-10:]
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
    print(f"    GET  /metrics   -> F1=0.298 | AUC=0.9475 | CV 3-fold par capteur")
    print(f"    GET  /sensors   -> 20 capteurs IFM")
    print(f"    GET  /anomalies -> Anomalies filtrees")
    print("=" * 80 + "\n")

    if not FASTAPI_OK:
        print("Installe les dépendances : pip install fastapi uvicorn pydantic scipy")
    else:
        port = int(os.environ.get("PORT", 8000))
        # TLS optionnel -- si TLS_CERT_FILE/TLS_KEY_FILE sont definis (voir
        # generate_selfsigned_cert.py pour un certificat de test), l'API sert
        # directement en HTTPS. Sinon comportement inchange (HTTP simple),
        # retro-compatible avec tous les deploiements existants.
        tls_cert = os.environ.get("TLS_CERT_FILE", "").strip()
        tls_key  = os.environ.get("TLS_KEY_FILE", "").strip()
        ssl_kwargs = {}
        if tls_cert and tls_key:
            if Path(tls_cert).exists() and Path(tls_key).exists():
                ssl_kwargs = {"ssl_certfile": tls_cert, "ssl_keyfile": tls_key}
                print(f"  TLS actif — HTTPS sur le port {port} (cert: {tls_cert})")
            else:
                print(f"  ATTENTION : TLS_CERT_FILE/TLS_KEY_FILE definis mais introuvables — HTTP sans chiffrement")
        uvicorn.run(app, host="0.0.0.0", port=port, reload=False, **ssl_kwargs)
