"""
train_rul_model.py
==================
Entraîne un modèle de régression ML dédié à la prédiction du RUL
(Remaining Useful Life) en heures.

Stratégie :
    Faute de données de défaillances réelles confirmées (aucun moteur n'a
    atteint la défaillance complète sur nov. 2025 → juin 2026), on construit
    des courbes de dégradation synthétiques réalistes à partir :
    1. Des données réelles full_data (distribution des features)
    2. D'un modèle de dégradation de Weibull paramétré sur les seuils industriels
    3. D'augmentation par injection de bruit calibré sur la variance réelle

Modèle :
    GradientBoostingRegressor (scikit-learn) — robuste aux outliers,
    pas de normalisation requise, interprétable via feature_importances_.

    Features d'entrée (25 temporelles + 20 spectrales = 45 features)
    Cible : rul_hours (0 → 2000 h)

Sorties :
    models/model_rul_v1.pkl    — modèle entraîné
    models/scaler_rul_v1.pkl   — RobustScaler
    models/metrics_rul_v1.json — métriques d'évaluation
    data/dataset_rul_synthetic.csv — dataset généré

Usage :
    python train_rul_model.py
    python train_rul_model.py --samples 5000 --test-size 0.2
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RUL-TRAIN] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("rul_train")

PROJECT_DIR  = Path(__file__).parent
MODEL_DIR    = PROJECT_DIR / "models"
DATA_DIR     = PROJECT_DIR / "data"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

MODEL_OUT   = MODEL_DIR / "model_rul_v1.pkl"
SCALER_OUT  = MODEL_DIR / "scaler_rul_v1.pkl"
METRICS_OUT = MODEL_DIR / "metrics_rul_v1.json"
DATASET_OUT = DATA_DIR  / "dataset_rul_synthetic.csv"

# Fréquence d'échantillonnage capteurs IFM (valeur estimée)
FS_SENSOR = 100.0   # Hz — valeur conservative pour les capteurs IFM VVB001
RPM_NOM   = 1450.0  # tr/min — vitesse nominale moteurs


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLE DE DÉGRADATION DE WEIBULL
# ══════════════════════════════════════════════════════════════════════════════

def weibull_degradation(t: np.ndarray, scale: float = 1000.0, shape: float = 2.5) -> np.ndarray:
    """
    Courbe de dégradation basée sur la distribution de Weibull.
    Retourne le score de dégradation normalisé [0, 1] à chaque instant t.

    shape > 1 : taux de défaillance croissant (usure)
    """
    return 1 - np.exp(-(t / scale) ** shape)


def generate_degradation_curve(
    total_life_hours: float,
    n_points: int = 50,
    noise_std: float = 0.02,
    weibull_shape: float = 2.5
) -> pd.DataFrame:
    """
    Génère une courbe de dégradation complète d'un moteur sur sa durée de vie.

    Retourne DataFrame avec :
        - t_hours        : Instant en heures
        - rul_hours      : Durée de vie restante (heures)
        - deg_score      : Score de dégradation [0, 1]
        - health_score   : Score de santé [0, 100]
        - vib_z_rms      : Vibration RMS Z simulée (mg)
        - vib_total      : Vibration totale 3D simulée (mg)
        - temp_mean      : Température simulée (°C)
        - current_mean   : Courant simulé (A)
        - ... (25 features temporelles + 20 spectrales)
    """
    rng = np.random.default_rng(int(total_life_hours * 7 + weibull_shape * 100))

    t = np.linspace(0, total_life_hours, n_points)
    deg = weibull_degradation(t, scale=total_life_hours * 0.8, shape=weibull_shape)
    deg += rng.normal(0, noise_std, n_points)
    deg = np.clip(deg, 0, 1)

    rul = total_life_hours - t

    # ── Features temporelles simulées ────────────────────────────────────────
    # Température : 35°C nominal → 65°C en fin de vie
    temp_mean = 35 + 30 * deg + rng.normal(0, 1.5, n_points)
    temp_mean = np.clip(temp_mean, 20, 80)
    temp_std  = 1.0 + 2.0 * deg + rng.exponential(0.3, n_points)

    # Vibration Z : 150 mg nominal → 1500 mg en fin de vie
    vib_z_base = 150 + 1350 * deg**1.5
    vib_z_mean = vib_z_base + rng.normal(0, 50, n_points)
    vib_z_mean = np.clip(vib_z_mean, 50, 2500)
    vib_z_std  = 20 + 100 * deg + rng.exponential(10, n_points)
    vib_z_rms  = vib_z_mean * (1 + 0.05 * rng.standard_normal(n_points))
    vib_z_crest = 2.5 + 6 * deg + rng.exponential(0.5, n_points)  # Facteur de crête
    vib_z_kurt  = 3.0 + 15 * deg**2 + rng.exponential(1, n_points)  # Kurtosis

    # Vibration X et Y (légèrement différentes de Z)
    vib_x_mean = vib_z_mean * (0.7 + 0.2 * rng.random(n_points))
    vib_y_mean = vib_z_mean * (0.6 + 0.3 * rng.random(n_points))
    vib_x_rms  = vib_x_mean * (1 + 0.03 * rng.standard_normal(n_points))
    vib_y_rms  = vib_y_mean * (1 + 0.03 * rng.standard_normal(n_points))

    # Vibration totale 3D
    vib_total = np.sqrt(vib_x_rms**2 + vib_y_rms**2 + vib_z_rms**2)

    # Courant moteur : légère augmentation en fin de vie (charge mécanique)
    current_mean = 10 + 5 * deg + rng.normal(0, 0.5, n_points)
    current_mean = np.clip(current_mean, 5, 30)
    current_std  = 0.5 + 1.0 * deg + rng.exponential(0.1, n_points)

    # Ratios inter-axes
    vib_xy_ratio = vib_x_mean / (vib_y_mean + 1e-9)
    vib_xz_ratio = vib_x_mean / (vib_z_mean + 1e-9)

    # Health score (composite)
    health_score = np.clip(100 - 65 * deg - 15 * deg**2 + rng.normal(0, 3, n_points), 5, 100)

    # Tendances (dérivée locale approximée)
    temp_trend = np.gradient(temp_mean)
    vib_trend  = np.gradient(vib_z_mean)

    # ── Features spectrales simulées ─────────────────────────────────────────
    # Plus le moteur se dégrade, plus :
    # - l'entropie spectrale diminue (énergie concentrée sur fréquences de défaut)
    # - le SNR BPFO/BPFI augmente
    # - la kurtosis d'enveloppe augmente
    spec_entropy    = 4.5 - 2.5 * deg + rng.normal(0, 0.2, n_points)
    env_kurtosis    = 3.0 + 12 * deg**1.5 + rng.exponential(0.5, n_points)
    env_crest       = 2.0 + 5 * deg + rng.exponential(0.3, n_points)
    bearing_bpfo_snr = 0.5 + 4.5 * deg**1.8 + rng.exponential(0.2, n_points)
    bearing_bpfi_snr = 0.3 + 2.5 * deg**2 + rng.exponential(0.15, n_points)
    bearing_bsf_snr  = 0.2 + 1.5 * deg**2.5 + rng.exponential(0.1, n_points)
    spec_centroid    = 200 + 300 * deg + rng.normal(0, 20, n_points)
    spec_bandwidth   = 150 + 200 * deg + rng.normal(0, 15, n_points)
    spec_band_low    = 0.4 - 0.2 * deg + rng.normal(0, 0.02, n_points)
    spec_band_mid    = 0.35 + 0.1 * deg + rng.normal(0, 0.02, n_points)
    spec_band_high   = 0.25 + 0.1 * deg + rng.normal(0, 0.02, n_points)
    wavelet_entropy  = 3.0 - 1.5 * deg + rng.normal(0, 0.1, n_points)
    harmonic_ratio   = 0.2 + 0.4 * deg + rng.normal(0, 0.03, n_points)
    noise_ratio      = 0.3 - 0.1 * deg + rng.normal(0, 0.02, n_points)
    spec_flatness    = 0.6 - 0.3 * deg + rng.normal(0, 0.02, n_points)
    spec_skewness    = 0.5 + 2.0 * deg + rng.normal(0, 0.1, n_points)
    spec_kurtosis_feat = 3.0 + 5.0 * deg + rng.normal(0, 0.2, n_points)
    bearing_severity = np.clip(np.round(3 * deg).astype(int), 0, 3)
    env_rms          = 20 + 200 * deg + rng.normal(0, 10, n_points)
    spec_total_energy = 1e4 + 1e6 * deg**1.5 + rng.exponential(1e3, n_points)

    # Accélération IFM (acc_p2p, acc_z2p, acc_crest, acc_rms)
    acc_p2p   = 2 * vib_z_mean * (1 + 0.1 * rng.random(n_points))
    acc_z2p   = vib_z_mean * (1 + 0.05 * rng.random(n_points))
    acc_crest_col = vib_z_crest * (1 + 0.1 * rng.random(n_points))
    acc_rms   = vib_z_rms * (0.9 + 0.1 * rng.random(n_points))

    df = pd.DataFrame({
        # Cible
        "rul_hours":    np.clip(rul, 0, total_life_hours),
        # Méta
        "deg_score":    deg,
        "total_life_h": total_life_hours,
        # Features temporelles (25)
        "temp_mean":    temp_mean,
        "temp_std":     temp_std,
        "temp_trend":   temp_trend,
        "temp_cur":     temp_mean + rng.normal(0, 0.5, n_points),
        "vib_z_mean":   vib_z_mean,
        "vib_z_std":    vib_z_std,
        "vib_z_rms_w":  vib_z_rms,
        "vib_z_kurt":   vib_z_kurt,
        "vib_z_crest":  vib_z_crest,
        "vib_z_cur":    vib_z_mean + rng.normal(0, 30, n_points),
        "vib_x_mean":   vib_x_mean,
        "vib_x_std":    np.abs(vib_z_std * 0.8 + rng.normal(0, 5, n_points)),
        "vib_x_rms_w":  vib_x_rms,
        "vib_x_kurt":   vib_z_kurt * 0.9 + rng.normal(0, 0.5, n_points),
        "vib_y_mean":   vib_y_mean,
        "vib_y_std":    np.abs(vib_z_std * 0.7 + rng.normal(0, 5, n_points)),
        "vib_y_rms_w":  vib_y_rms,
        "vib_y_kurt":   vib_z_kurt * 0.85 + rng.normal(0, 0.5, n_points),
        "vib_xy_ratio": vib_xy_ratio,
        "vib_xz_ratio": vib_xz_ratio,
        "current_mean": current_mean,
        "current_std":  current_std,
        "vib_total":    vib_total,
        "health_score": health_score,
        "acc_p2p":      acc_p2p,
        "acc_z2p":      acc_z2p,
        "acc_crest":    acc_crest_col,
        "acc_rms":      acc_rms,
        # Features spectrales (20)
        "spec_peak_freq":       spec_centroid * 0.7 + rng.normal(0, 5, n_points),
        "spec_centroid":        spec_centroid,
        "spec_bandwidth":       spec_bandwidth,
        "spec_entropy":         spec_entropy,
        "spec_flatness":        spec_flatness,
        "spec_total_energy":    spec_total_energy,
        "spec_band_low_ratio":  np.clip(spec_band_low, 0, 1),
        "spec_band_mid_ratio":  np.clip(spec_band_mid, 0, 1),
        "spec_band_high_ratio": np.clip(spec_band_high, 0, 1),
        "env_kurtosis":         env_kurtosis,
        "env_rms":              env_rms,
        "env_crest":            env_crest,
        "bearing_bpfo_snr":     bearing_bpfo_snr,
        "bearing_bpfi_snr":     bearing_bpfi_snr,
        "bearing_bsf_snr":      bearing_bsf_snr,
        "bearing_fault_severity": bearing_severity.astype(float),
        "spectral_skewness":    spec_skewness,
        "spectral_kurtosis":    spec_kurtosis_feat,
        "harmonic_ratio":       np.clip(harmonic_ratio, 0, 1),
        "noise_ratio":          np.clip(noise_ratio, 0, 1),
        "wavelet_entropy":      wavelet_entropy,
    })

    return df


def generate_training_dataset(
    n_motors: int = 150,
    points_per_motor: int = 50,
    noise_std: float = 0.03
) -> pd.DataFrame:
    """
    Génère un dataset d'entraînement complet pour le modèle RUL.

    Simule n_motors moteurs avec des durées de vie variées (500h → 3000h)
    et des profils de dégradation différents (Weibull shape 1.5 → 3.5).
    """
    log.info(f"Génération du dataset : {n_motors} moteurs × {points_per_motor} points = {n_motors * points_per_motor} lignes")

    dfs = []
    motor_lifetimes = np.random.default_rng(42).uniform(500, 3000, n_motors)
    weibull_shapes  = np.random.default_rng(43).uniform(1.5, 3.5, n_motors)

    for i in range(n_motors):
        life_h = float(motor_lifetimes[i])
        shape  = float(weibull_shapes[i])
        noise  = noise_std * (0.5 + 0.5 * np.random.random())

        df_motor = generate_degradation_curve(
            total_life_hours=life_h,
            n_points=points_per_motor,
            noise_std=noise,
            weibull_shape=shape
        )
        df_motor["motor_id"] = f"SIM_MOTOR_{i:04d}"
        dfs.append(df_motor)

        if (i + 1) % 25 == 0:
            log.info(f"  {i+1}/{n_motors} moteurs générés...")

    dataset = pd.concat(dfs, ignore_index=True)

    # Charger les données réelles si disponibles pour affiner les distributions
    real_path = DATA_DIR / "dataset_real_full.csv"
    if real_path.exists():
        try:
            df_real = pd.read_csv(real_path, nrows=2000)
            log.info(f"Données réelles chargées : {len(df_real)} lignes — calibration des distributions")

            # Ajuster les moyennes et écarts-types sur les données réelles
            for col in ["temp_mean", "vib_z_mean", "current_mean"]:
                if col in df_real.columns and col in dataset.columns:
                    real_mean = df_real[col].dropna().mean()
                    real_std  = df_real[col].dropna().std()
                    sim_mean  = dataset[col].mean()
                    sim_std   = dataset[col].std()
                    if sim_std > 0 and real_std > 0:
                        dataset[col] = (dataset[col] - sim_mean) / sim_std * real_std + real_mean
        except Exception as e:
            log.warning(f"Calibration données réelles échouée : {e}")

    log.info(f"Dataset généré : {len(dataset)} lignes, {dataset.shape[1]} colonnes")
    log.info(f"  RUL min/max : {dataset['rul_hours'].min():.0f} / {dataset['rul_hours'].max():.0f} h")
    log.info(f"  Moyenne RUL : {dataset['rul_hours'].mean():.0f} h")

    return dataset


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    # Temporelles (25)
    "temp_mean", "temp_std", "temp_trend", "temp_cur",
    "vib_z_mean", "vib_z_std", "vib_z_rms_w", "vib_z_kurt", "vib_z_crest", "vib_z_cur",
    "vib_x_mean", "vib_x_std", "vib_x_rms_w", "vib_x_kurt",
    "vib_y_mean", "vib_y_std", "vib_y_rms_w", "vib_y_kurt",
    "vib_xy_ratio", "vib_xz_ratio",
    "current_mean", "current_std",
    "vib_total", "health_score",
    "acc_p2p",
    # Spectrales (21)
    "spec_peak_freq", "spec_centroid", "spec_bandwidth", "spec_entropy",
    "spec_flatness", "spec_total_energy",
    "spec_band_low_ratio", "spec_band_mid_ratio", "spec_band_high_ratio",
    "env_kurtosis", "env_rms", "env_crest",
    "bearing_bpfo_snr", "bearing_bpfi_snr", "bearing_bsf_snr",
    "bearing_fault_severity",
    "spectral_skewness", "spectral_kurtosis",
    "harmonic_ratio", "noise_ratio",
    "wavelet_entropy",
]

TARGET_COL = "rul_hours"


def train_rul_model(dataset: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Entraîne le modèle GradientBoostingRegressor pour la prédiction RUL.

    Returns métriques d'évaluation.
    """
    # Nettoyage
    available_features = [c for c in FEATURE_COLS if c in dataset.columns]
    log.info(f"Features utilisées : {len(available_features)}/{len(FEATURE_COLS)}")

    df_clean = dataset[available_features + [TARGET_COL]].dropna()
    df_clean = df_clean[df_clean[TARGET_COL] >= 0]

    X = df_clean[available_features].values
    y = df_clean[TARGET_COL].values

    log.info(f"Dataset nettoyé : {len(X)} lignes")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Scaler
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Modèle principal : GradientBoosting
    log.info("Entraînement GradientBoostingRegressor...")
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )
    gbr.fit(X_train_s, y_train)

    # Évaluation
    y_pred_train = gbr.predict(X_train_s)
    y_pred_test  = gbr.predict(X_test_s)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test  = mean_absolute_error(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2_test   = r2_score(y_test, y_pred_test)

    # Score relatif : MAE en % de la moyenne RUL
    mean_rul  = float(np.mean(y_test))
    mae_pct   = (mae_test / (mean_rul + 1e-9)) * 100

    log.info(f"  MAE  train : {mae_train:.1f} h")
    log.info(f"  MAE  test  : {mae_test:.1f} h  ({mae_pct:.1f}% de la moyenne RUL)")
    log.info(f"  RMSE test  : {rmse_test:.1f} h")
    log.info(f"  R²   test  : {r2_test:.4f}")

    # Validation croisée (5-fold)
    log.info("Validation croisée 5-fold...")
    cv_scores = cross_val_score(gbr, X_train_s, y_train, cv=5,
                                scoring="neg_mean_absolute_error")
    cv_mae = float(-np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    log.info(f"  CV MAE : {cv_mae:.1f} ± {cv_std:.1f} h")

    # Modèle de secours : RandomForest (plus léger pour l'Edge)
    log.info("Entraînement RandomForestRegressor (modèle Edge léger)...")
    rfr = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rfr.fit(X_train_s, y_train)
    mae_rf = mean_absolute_error(y_test, rfr.predict(X_test_s))
    r2_rf  = r2_score(y_test, rfr.predict(X_test_s))
    log.info(f"  RF MAE test : {mae_rf:.1f} h | R² : {r2_rf:.4f}")

    # Feature importances top 10
    importances = dict(zip(available_features, gbr.feature_importances_))
    top10 = sorted(importances.items(), key=lambda x: -x[1])[:10]
    log.info("  Top 10 features :")
    for name, imp in top10:
        log.info(f"    {name:<35} : {imp:.4f}")

    # Sauvegarde
    joblib.dump(gbr,    MODEL_OUT,  compress=3)
    joblib.dump(scaler, SCALER_OUT, compress=3)

    # Sauvegarder aussi le RF pour l'edge
    rf_path = MODEL_DIR / "model_rul_rf_edge_v1.pkl"
    joblib.dump(rfr, rf_path, compress=3)

    # Sauvegarder la liste des features
    feat_path = MODEL_DIR / "features_rul_v1.pkl"
    joblib.dump(available_features, feat_path)

    log.info(f"Modèle GBR sauvegardé → {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1024:.0f} KB)")
    log.info(f"Modèle RF  sauvegardé → {rf_path} ({rf_path.stat().st_size / 1024:.0f} KB)")

    metrics = {
        "model":          "GradientBoostingRegressor",
        "n_features":     len(available_features),
        "n_train":        int(len(X_train)),
        "n_test":         int(len(X_test)),
        "mae_train_h":    round(mae_train, 2),
        "mae_test_h":     round(mae_test, 2),
        "mae_test_pct":   round(mae_pct, 2),
        "rmse_test_h":    round(rmse_test, 2),
        "r2_test":        round(r2_test, 4),
        "cv_mae_h":       round(cv_mae, 2),
        "cv_std_h":       round(cv_std, 2),
        "rf_mae_test_h":  round(mae_rf, 2),
        "rf_r2_test":     round(r2_rf, 4),
        "top10_features": [{"feature": n, "importance": round(v, 4)} for n, v in top10],
        "trained_at":     pd.Timestamp.now().isoformat(),
        "feature_cols":   available_features,
    }

    METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info(f"Métriques sauvegardées → {METRICS_OUT}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  INFÉRENCE — utilisation depuis l'API
# ══════════════════════════════════════════════════════════════════════════════

class RULPredictor:
    """
    Wrapper pour l'inférence RUL depuis l'API FastAPI.
    Charge le modèle GBR et scaler une seule fois au démarrage.

    Usage :
        predictor = RULPredictor()
        result = predictor.predict(features_dict)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return True
        try:
            self.model   = joblib.load(MODEL_OUT)
            self.scaler  = joblib.load(SCALER_OUT)
            self.features = joblib.load(MODEL_DIR / "features_rul_v1.pkl")
            self._loaded = True
            log.info(f"RULPredictor chargé : {len(self.features)} features")
            return True
        except FileNotFoundError:
            log.warning("Modèle RUL non trouvé — lance : python train_rul_model.py")
            self._loaded = False
            return False

    def predict(self, features_dict: dict) -> dict:
        """
        Prédit le RUL en heures à partir d'un dictionnaire de features.

        Args:
            features_dict : Dictionnaire feature_name → valeur

        Returns dict avec rul_hours, confidence, alert_level, recommendation
        """
        if not self._loaded and not self.load():
            return self._fallback_heuristic(features_dict)

        try:
            # Construire le vecteur de features dans l'ordre attendu
            x = np.array([features_dict.get(f, 0.0) for f in self.features]).reshape(1, -1)

            # Remplacer NaN par 0
            x = np.nan_to_num(x, nan=0.0)

            x_scaled = self.scaler.transform(x)
            rul_raw   = float(self.model.predict(x_scaled)[0])
            rul_hours = max(0.0, round(rul_raw, 1))
            rul_days  = round(rul_hours / 24.0, 2)

            # Niveau de confiance basé sur le health_score
            health = features_dict.get("health_score", 50.0)
            if health is None or np.isnan(health):
                health = 50.0

            if health >= 80:
                confidence = "HAUTE"
            elif health >= 60:
                confidence = "MOYENNE"
            else:
                confidence = "FAIBLE"

            # Niveau d'alerte
            if rul_hours > 500:
                alert_level = "OK"
                recommendation = "Fonctionnement normal. Prochaine inspection planifiée."
            elif rul_hours > 100:
                alert_level = "ATTENTION"
                recommendation = f"Planifier une maintenance préventive dans les {int(rul_days)} jours."
            elif rul_hours > 24:
                alert_level = "URGENT"
                recommendation = f"Maintenance requise sous {int(rul_hours)} heures. Préparer les pièces de rechange."
            else:
                alert_level = "CRITIQUE"
                recommendation = f"Arrêt imminent ! RUL estimé : {rul_hours:.1f}h. Arrêter le moteur dès que possible."

            return {
                "rul_hours":    rul_hours,
                "rul_days":     rul_days,
                "confidence":   confidence,
                "alert_level":  alert_level,
                "recommendation": recommendation,
                "model_type":   "GradientBoosting_v1",
            }

        except Exception as e:
            log.error(f"Erreur inférence RUL : {e}")
            return self._fallback_heuristic(features_dict)

    def _fallback_heuristic(self, features_dict: dict) -> dict:
        """Heuristique de secours si le modèle n'est pas disponible."""
        health = features_dict.get("health_score", 50.0) or 50.0
        rul_hours = max(0.0, health / 100 * 1000)
        return {
            "rul_hours":    round(rul_hours, 1),
            "rul_days":     round(rul_hours / 24, 2),
            "confidence":   "FAIBLE",
            "alert_level":  "OK" if rul_hours > 500 else "ATTENTION",
            "recommendation": "Modèle RUL non disponible — estimation par heuristique.",
            "model_type":   "heuristic_fallback",
        }


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Entraîne le modèle RUL ML dédié")
    parser.add_argument("--samples",   type=int,   default=150,  help="Nombre de moteurs simulés (défaut 150)")
    parser.add_argument("--points",    type=int,   default=50,   help="Points par moteur (défaut 50)")
    parser.add_argument("--test-size", type=float, default=0.2,  help="Fraction test (défaut 0.2)")
    parser.add_argument("--no-save",   action="store_true",      help="Ne pas sauvegarder le dataset CSV")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  Entraînement Modèle RUL — Maintenance Prédictive")
    log.info("=" * 60)

    # Génération du dataset
    dataset = generate_training_dataset(
        n_motors=args.samples,
        points_per_motor=args.points
    )

    if not args.no_save:
        dataset.to_csv(DATASET_OUT, index=False, encoding="utf-8")
        log.info(f"Dataset sauvegardé → {DATASET_OUT}")

    # Entraînement
    metrics = train_rul_model(dataset, test_size=args.test_size)

    log.info("\n" + "=" * 60)
    log.info("  RÉSUMÉ")
    log.info("=" * 60)
    log.info(f"  MAE test  : {metrics['mae_test_h']:.1f} h  ({metrics['mae_test_pct']:.1f}%)")
    log.info(f"  RMSE test : {metrics['rmse_test_h']:.1f} h")
    log.info(f"  R²        : {metrics['r2_test']:.4f}")
    log.info(f"  Modèle    → {MODEL_OUT}")
    log.info(f"  Métriques → {METRICS_OUT}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
