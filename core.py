"""
core.py
=======
État partagé et logique métier de l'API Maintenance Prédictive : chargement des
modèles, extraction de features, ensemble ML, calcul RUL, persistance de
l'historique. Aucune route FastAPI ici — uniquement ce dont les routers
(voir routers/) ont besoin pour fonctionner.

⚠️  Piège d'import à connaître si tu ajoutes du code ici :
Les routers font `import core` et lisent `core.xxx` (jamais
`from core import xxx` pour un nom réassigné par load_all_models(), comme
scaler/pca/features_list/thresholds/meta_lr/df_results) -- ces variables sont
None/vides à l'import, puis RÉASSIGNÉES (pas mutées en place) au démarrage de
l'API (lifespan -> load_all_models()). Un `from core import scaler` figerait
la valeur None capturée à l'import, avant que load_all_models() ne tourne.
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
from datetime import datetime
from collections import deque
from scipy.stats import entropy as sp_entropy

# ── Chargement .env (ignoré silencieusement si absent — Docker injecte les vars) ─
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("pydantic non installé. Lance : pip install fastapi uvicorn pydantic")

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

# Cache des derniers résultats par capteur (pour /v1/results)
_latest_results: dict = {}

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

IFM_KNOWN_IDS = {
    "07da47b8","0ff416d2","2c6254af","3a782f1b","4b5e4b32",
    "53cb61b2","68c11f06","6e0c1740","718fd2af","8f7f2f7e",
    "91d92804","99695e98","a6a46be1","aa7b02a1","b2acdf45",
    "bc59bf5f","d9508e77","eb084747","f48c25f9","ed6fa322",
}


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
        _latest_results.pop(sid, None)
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
