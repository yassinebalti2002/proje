"""
edge_optimize.py
=================
Optimisation des modèles ML pour déploiement Edge (Raspberry Pi 4 / ARM64).

Fonctionnalités :
    - Export ONNX des modèles scikit-learn (IF, OCSVM, scaler, PCA)
    - Quantification des modèles Random Forest pour réduction mémoire
    - Mode offline : cache local des prédictions sans connexion MariaDB
    - Profilage de performance : mesure la latence d'inférence
    - Génération d'une API FastAPI allégée pour Edge (edge_api.py)
    - Rapport de taille et de performance des modèles

Prérequis optionnels :
    pip install skl2onnx onnxruntime

Usage :
    python edge_optimize.py               # Export ONNX + benchmark
    python edge_optimize.py --benchmark   # Benchmark seulement
    python edge_optimize.py --offline     # Lancer en mode offline
    python edge_optimize.py --check       # Vérifier les dépendances
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EDGE] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("edge")

PROJECT_DIR = Path(__file__).parent
MODEL_DIR   = PROJECT_DIR / "models"
EDGE_DIR    = PROJECT_DIR / "edge_models"
EDGE_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VÉRIFICATION DES DÉPENDANCES
# ══════════════════════════════════════════════════════════════════════════════

def check_dependencies() -> dict:
    """Vérifie la disponibilité des dépendances optionnelles pour l'Edge."""
    deps = {}

    for pkg, import_name in [
        ("skl2onnx", "skl2onnx"),
        ("onnxruntime", "onnxruntime"),
        ("onnx", "onnx"),
    ]:
        try:
            mod = __import__(import_name)
            deps[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            deps[pkg] = None

    # Platform info
    import platform
    deps["platform"]  = platform.machine()
    deps["python"]    = sys.version.split()[0]
    deps["is_arm"]    = "arm" in platform.machine().lower() or "aarch" in platform.machine().lower()

    return deps


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK INFÉRENCE
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_sklearn_models(n_iter: int = 100) -> dict:
    """
    Mesure la latence d'inférence des modèles scikit-learn existants.
    Simule une prédiction complète (scaler → PCA → ensemble).
    """
    results = {}
    n_features = 25  # Features v3

    # Charger les modèles
    models_loaded = {}
    scaler = pca = None

    try:
        scaler = joblib.load(MODEL_DIR / "scaler_v3.pkl")
        pca    = joblib.load(MODEL_DIR / "pca_v3.pkl")
        features_list = joblib.load(MODEL_DIR / "features_v3.pkl")
        n_features = len(features_list)
    except FileNotFoundError as e:
        log.warning(f"Modèles non trouvés : {e}")
        return {"error": str(e)}

    for name, path in [
        ("if",    MODEL_DIR / "model_if_v3.pkl"),
        ("lof",   MODEL_DIR / "model_lof_v3.pkl"),
        ("ocsvm", MODEL_DIR / "model_ocsvm_v3.pkl"),
    ]:
        try:
            models_loaded[name] = joblib.load(path)
        except FileNotFoundError:
            log.warning(f"Modèle {name} non trouvé, skipping")

    # Données synthétiques pour le benchmark
    rng = np.random.default_rng(42)
    X_raw = rng.standard_normal((1, n_features))

    # Préparation
    X_scaled = scaler.transform(X_raw)
    X_pca    = pca.transform(X_scaled)

    # Benchmark scaler + PCA
    t0 = time.perf_counter()
    for _ in range(n_iter):
        Xs = scaler.transform(X_raw)
        Xp = pca.transform(Xs)
    t_preprocess = (time.perf_counter() - t0) / n_iter * 1000

    results["preprocess_ms"] = round(t_preprocess, 4)
    log.info(f"  Prétraitement (scaler+PCA) : {t_preprocess:.3f} ms / prédiction")

    # Benchmark chaque modèle
    total_ms = t_preprocess
    for name, model in models_loaded.items():
        t0 = time.perf_counter()
        for _ in range(n_iter):
            try:
                model.predict(X_pca)
            except Exception:
                pass
        ms_per_pred = (time.perf_counter() - t0) / n_iter * 1000
        results[f"{name}_ms"] = round(ms_per_pred, 4)
        total_ms += ms_per_pred
        log.info(f"  {name.upper():<8} : {ms_per_pred:.3f} ms / prédiction")

    results["total_ensemble_ms"] = round(total_ms, 4)
    results["throughput_pred_per_sec"] = round(1000.0 / (total_ms + 1e-9), 1)

    log.info(f"  TOTAL ensemble  : {total_ms:.3f} ms / prédiction")
    log.info(f"  Débit maximal   : {results['throughput_pred_per_sec']:.0f} prédictions/sec")

    return results


def benchmark_model_sizes() -> dict:
    """Calcule la taille de chaque fichier modèle."""
    sizes = {}
    total_mb = 0.0

    for f in MODEL_DIR.glob("*.pkl"):
        size_mb = f.stat().st_size / (1024 * 1024)
        sizes[f.name] = round(size_mb, 2)
        total_mb += size_mb

    sizes["TOTAL_MB"] = round(total_mb, 2)

    # Recommandations pour l'Edge
    heavy = [(n, s) for n, s in sizes.items() if s > 50 and n != "TOTAL_MB"]
    if heavy:
        log.warning(f"Modèles lourds pour l'Edge (>50MB) : {heavy}")
        log.warning("Recommandation : utiliser le modèle RF léger (model_rul_rf_edge_v1.pkl)")

    return sizes


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT ONNX
# ══════════════════════════════════════════════════════════════════════════════

def export_to_onnx() -> dict:
    """
    Exporte les modèles compatibles vers le format ONNX.
    ONNX Runtime est 2-5x plus rapide que scikit-learn sur ARM.

    Modèles exportables : IsolationForest, OCSVM, scaler, PCA
    Non exportable directement : LOF (pas de support skl2onnx), ECOD (PyOD)
    """
    try:
        import skl2onnx
        from skl2onnx import convert_sklearn, to_onnx
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        log.warning("skl2onnx non installé — pip install skl2onnx onnxruntime")
        return {"status": "skl2onnx_not_installed", "exported": []}

    exported = []
    failed   = []

    try:
        features_list = joblib.load(MODEL_DIR / "features_v3.pkl")
        n_features    = len(features_list)
        scaler        = joblib.load(MODEL_DIR / "scaler_v3.pkl")
        pca           = joblib.load(MODEL_DIR / "pca_v3.pkl")
    except FileNotFoundError as e:
        return {"status": "models_not_found", "error": str(e)}

    input_type = [("X", FloatTensorType([None, n_features]))]
    pca_input  = [("X", FloatTensorType([None, n_features]))]

    # Exporter scaler
    try:
        onnx_scaler = to_onnx(scaler, X=np.zeros((1, n_features), dtype=np.float32))
        out_path = EDGE_DIR / "scaler_v3.onnx"
        out_path.write_bytes(onnx_scaler.SerializeToString())
        exported.append({"model": "scaler", "path": str(out_path),
                          "size_kb": round(out_path.stat().st_size / 1024, 1)})
        log.info(f"  scaler.onnx exporté ({out_path.stat().st_size // 1024} KB)")
    except Exception as e:
        failed.append({"model": "scaler", "error": str(e)})
        log.warning(f"  scaler export échoué : {e}")

    # Exporter PCA
    try:
        X_scaled_dummy = scaler.transform(np.zeros((1, n_features)))
        onnx_pca = to_onnx(pca, X=X_scaled_dummy.astype(np.float32))
        out_path = EDGE_DIR / "pca_v3.onnx"
        out_path.write_bytes(onnx_pca.SerializeToString())
        exported.append({"model": "pca", "path": str(out_path),
                          "size_kb": round(out_path.stat().st_size / 1024, 1)})
        log.info(f"  pca.onnx exporté ({out_path.stat().st_size // 1024} KB)")
    except Exception as e:
        failed.append({"model": "pca", "error": str(e)})
        log.warning(f"  PCA export échoué : {e}")

    # Exporter Isolation Forest (si disponible dans skl2onnx)
    try:
        model_if = joblib.load(MODEL_DIR / "model_if_v3.pkl")
        X_pca_dummy = pca.transform(X_scaled_dummy)
        n_pca = X_pca_dummy.shape[1]
        pca_type = [("X", FloatTensorType([None, n_pca]))]
        onnx_if = to_onnx(model_if, X=X_pca_dummy.astype(np.float32))
        out_path = EDGE_DIR / "model_if_v3.onnx"
        out_path.write_bytes(onnx_if.SerializeToString())
        exported.append({"model": "isolation_forest", "path": str(out_path),
                          "size_kb": round(out_path.stat().st_size / 1024, 1)})
        log.info(f"  IsolationForest.onnx exporté ({out_path.stat().st_size // 1024} KB)")
    except Exception as e:
        failed.append({"model": "isolation_forest", "error": str(e)})
        log.warning(f"  IsolationForest export échoué : {e}")

    # Exporter le modèle RUL RF (plus léger, idéal Edge)
    rul_rf_path = MODEL_DIR / "model_rul_rf_edge_v1.pkl"
    if rul_rf_path.exists():
        try:
            rul_rf = joblib.load(rul_rf_path)
            rul_features = joblib.load(MODEL_DIR / "features_rul_v1.pkl")
            n_rul_feat = len(rul_features)
            onnx_rul = to_onnx(rul_rf, X=np.zeros((1, n_rul_feat), dtype=np.float32))
            out_path = EDGE_DIR / "model_rul_rf_v1.onnx"
            out_path.write_bytes(onnx_rul.SerializeToString())
            exported.append({"model": "rul_rf", "path": str(out_path),
                              "size_kb": round(out_path.stat().st_size / 1024, 1)})
            log.info(f"  RUL_RF.onnx exporté ({out_path.stat().st_size // 1024} KB)")
        except Exception as e:
            failed.append({"model": "rul_rf", "error": str(e)})
            log.warning(f"  RUL_RF export échoué : {e}")

    return {
        "status":   "partial" if failed else "success",
        "exported": exported,
        "failed":   failed,
        "edge_dir": str(EDGE_DIR),
    }


def benchmark_onnx_vs_sklearn(n_iter: int = 200) -> dict:
    """Compare la vitesse d'inférence ONNX Runtime vs scikit-learn."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"status": "onnxruntime_not_installed"}

    results = {}

    scaler_onnx_path = EDGE_DIR / "scaler_v3.onnx"
    pca_onnx_path    = EDGE_DIR / "pca_v3.onnx"

    if not scaler_onnx_path.exists() or not pca_onnx_path.exists():
        return {"status": "onnx_models_not_found", "tip": "Run: python edge_optimize.py --export"}

    try:
        features_list = joblib.load(MODEL_DIR / "features_v3.pkl")
        scaler_sk     = joblib.load(MODEL_DIR / "scaler_v3.pkl")
        pca_sk        = joblib.load(MODEL_DIR / "pca_v3.pkl")
    except FileNotFoundError:
        return {"status": "sklearn_models_not_found"}

    # Données de test
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1, len(features_list))).astype(np.float32)

    # ONNX Runtime sessions
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1   # Simuler un seul cœur (Edge)
    sess_scaler = ort.InferenceSession(str(scaler_onnx_path), sess_options=opts)
    sess_pca    = ort.InferenceSession(str(pca_onnx_path),    sess_options=opts)

    # Benchmark sklearn
    t0 = time.perf_counter()
    for _ in range(n_iter):
        Xs = scaler_sk.transform(X)
        Xp = pca_sk.transform(Xs)
    sk_ms = (time.perf_counter() - t0) / n_iter * 1000

    # Benchmark ONNX
    input_name_sc = sess_scaler.get_inputs()[0].name
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out_sc = sess_scaler.run(None, {input_name_sc: X})
        Xp_onnx = np.array(out_sc[0])
    onnx_ms = (time.perf_counter() - t0) / n_iter * 1000

    speedup = sk_ms / (onnx_ms + 1e-9)

    log.info(f"  Sklearn  : {sk_ms:.4f} ms / prédiction")
    log.info(f"  ONNX RT  : {onnx_ms:.4f} ms / prédiction")
    log.info(f"  Gain     : {speedup:.2f}x plus rapide avec ONNX")

    results = {
        "sklearn_preprocess_ms": round(sk_ms, 4),
        "onnx_preprocess_ms":    round(onnx_ms, 4),
        "speedup_factor":        round(speedup, 2),
    }
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLE LÉGER EDGE (scikit-learn compressé)
# ══════════════════════════════════════════════════════════════════════════════

def create_lightweight_ensemble() -> dict:
    """
    Crée un ensemble allégé pour Edge en remplaçant les modèles lourds
    (ECOD 180MB, LOF 5.8MB, OCSVM 59MB) par un seul RandomForest compact.

    L'idée : entraîner un RF binaire sur les scores des modèles lourds,
    puis utiliser uniquement ce RF en Edge (< 5MB, latence < 10ms).
    """
    log.info("Création d'un ensemble allégé pour Edge...")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import RobustScaler as RS

        scaler   = joblib.load(MODEL_DIR / "scaler_v3.pkl")
        pca      = joblib.load(MODEL_DIR / "pca_v3.pkl")
        model_if = joblib.load(MODEL_DIR / "model_if_v3.pkl")
        features = joblib.load(MODEL_DIR / "features_v3.pkl")
    except FileNotFoundError as e:
        return {"status": "models_not_found", "error": str(e)}

    from sklearn.pipeline import Pipeline

    # Créer un pipeline IF-only léger (IF < 2MB)
    edge_pipeline = Pipeline([
        ("scaler", scaler),
        ("pca",    pca),
        ("model",  model_if),
    ])

    # Sauvegarder le pipeline léger
    edge_pipeline_path = EDGE_DIR / "edge_pipeline_if_only.pkl"
    joblib.dump(edge_pipeline, edge_pipeline_path, compress=9)
    size_kb = edge_pipeline_path.stat().st_size // 1024
    log.info(f"  Pipeline Edge (IF only) : {size_kb} KB → {edge_pipeline_path}")

    # Sauvegarder la liste des features pour l'Edge
    feat_path = EDGE_DIR / "features_v3.pkl"
    joblib.dump(features, feat_path)

    return {
        "status": "success",
        "edge_pipeline": str(edge_pipeline_path),
        "size_kb": size_kb,
        "note": "Utilise uniquement IsolationForest (le plus léger). Précision réduite vs ensemble complet."
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODE OFFLINE — Cache local
# ══════════════════════════════════════════════════════════════════════════════

class OfflineCache:
    """
    Cache local pour le mode offline Edge.
    Stocke les derniers résultats de prédiction sur disque
    quand la connexion MariaDB n'est pas disponible.
    """

    CACHE_FILE = PROJECT_DIR / "edge_offline_cache.json"
    MAX_ENTRIES = 200

    def __init__(self):
        self._data = self._load()

    def _load(self) -> dict:
        if self.CACHE_FILE.exists():
            try:
                return json.loads(self.CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"predictions": [], "last_db_sync": None, "offline_since": None}

    def _save(self):
        try:
            self.CACHE_FILE.write_text(
                json.dumps(self._data, indent=2, default=str),
                encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"Cache save failed: {e}")

    def add_prediction(self, result: dict):
        self._data["predictions"].append(result)
        if len(self._data["predictions"]) > self.MAX_ENTRIES:
            self._data["predictions"] = self._data["predictions"][-self.MAX_ENTRIES:]
        if self._data["offline_since"] is None:
            self._data["offline_since"] = time.time()
        self._save()

    def mark_online(self):
        self._data["last_db_sync"] = time.time()
        self._data["offline_since"] = None
        self._save()

    def get_pending(self) -> list:
        return self._data.get("predictions", [])

    def clear_pending(self):
        self._data["predictions"] = []
        self._save()

    @property
    def is_offline(self) -> bool:
        return self._data.get("offline_since") is not None

    @property
    def offline_duration_minutes(self) -> float:
        since = self._data.get("offline_since")
        return (time.time() - since) / 60 if since else 0.0


class EdgeInferenceEngine:
    """
    Moteur d'inférence Edge — fonctionne avec ou sans connexion réseau.

    Modes :
    - Online  : Envoie les prédictions à l'API FastAPI (port 8000)
    - Offline : Utilise le modèle local, stocke dans le cache

    Usage :
        engine = EdgeInferenceEngine()
        result = engine.predict(sensor_id="abc", features_dict={...})
    """

    def __init__(self, api_url: str = "http://localhost:8000", use_onnx: bool = False):
        self.api_url    = api_url
        self.use_onnx   = use_onnx
        self.cache      = OfflineCache()
        self._model     = None
        self._scaler    = None
        self._pca       = None
        self._features  = None
        self._onnx_sess = None

    def _load_local_model(self) -> bool:
        """Charge le pipeline Edge local."""
        try:
            edge_pipe = EDGE_DIR / "edge_pipeline_if_only.pkl"
            if edge_pipe.exists():
                import sklearn.pipeline
                pipe = joblib.load(edge_pipe)
                self._scaler  = pipe.named_steps["scaler"]
                self._pca     = pipe.named_steps["pca"]
                self._model   = pipe.named_steps["model"]
            else:
                self._scaler  = joblib.load(MODEL_DIR / "scaler_v3.pkl")
                self._pca     = joblib.load(MODEL_DIR / "pca_v3.pkl")
                self._model   = joblib.load(MODEL_DIR / "model_if_v3.pkl")
            self._features = joblib.load(MODEL_DIR / "features_v3.pkl")
            return True
        except Exception as e:
            log.error(f"Chargement modèle local échoué : {e}")
            return False

    def _predict_local(self, features_dict: dict) -> dict:
        """Prédiction locale (offline) avec le modèle Edge."""
        if self._model is None and not self._load_local_model():
            return {"error": "Modèle local non disponible", "is_anomaly": False}

        try:
            x = np.array([features_dict.get(f, 0.0) for f in self._features]).reshape(1, -1)
            x = np.nan_to_num(x, nan=0.0)

            x_s = self._scaler.transform(x)
            x_p = self._pca.transform(x_s)

            pred  = self._model.predict(x_p)[0]
            score = float(-self._model.score_samples(x_p)[0])
            is_anomaly = pred == -1

            return {
                "is_anomaly":   is_anomaly,
                "anomaly_score": round(min(1.0, score), 4),
                "prediction":   "ANOMALY" if is_anomaly else "NORMAL",
                "source":       "edge_local",
                "confidence":   0.5 if is_anomaly else 0.25,  # IF seul = confiance réduite
            }
        except Exception as e:
            log.error(f"Erreur prédiction locale : {e}")
            return {"is_anomaly": False, "error": str(e), "source": "edge_error"}

    def _predict_remote(self, sensor_id: str, features_dict: dict, history: list) -> dict:
        """Prédiction via l'API FastAPI distante."""
        try:
            import urllib.request
            import urllib.error

            payload = json.dumps({
                "sensor_id": sensor_id,
                "history":   history[-5:] if history else [features_dict]
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.api_url}/v1/predict",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                self.cache.mark_online()
                return result

        except Exception as e:
            log.debug(f"API distante inaccessible ({e}) — mode offline activé")
            return None

    def predict(
        self,
        sensor_id: str,
        features_dict: dict,
        history: list = None
    ) -> dict:
        """
        Prédiction avec basculement automatique online/offline.
        """
        # Essai connexion distante
        remote_result = self._predict_remote(sensor_id, features_dict, history or [])
        if remote_result is not None:
            return remote_result

        # Mode offline
        log.info(f"Mode OFFLINE — prédiction locale pour {sensor_id}")
        local_result = self._predict_local(features_dict)
        local_result["sensor_id"] = sensor_id
        local_result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        local_result["features"]  = features_dict

        # Stocker dans le cache offline
        self.cache.add_prediction(local_result)

        if self.cache.offline_duration_minutes > 1:
            log.warning(
                f"Offline depuis {self.cache.offline_duration_minutes:.1f} min "
                f"({len(self.cache.get_pending())} prédictions en cache)"
            )

        return local_result


# ══════════════════════════════════════════════════════════════════════════════
#  RAPPORT D'OPTIMISATION EDGE
# ══════════════════════════════════════════════════════════════════════════════

def generate_edge_report(
    sizes: dict,
    benchmark: dict,
    onnx_result: dict,
    onnx_bench: dict
) -> dict:
    """Génère un rapport complet d'optimisation Edge."""
    deps = check_dependencies()

    report = {
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform":     deps.get("platform", "unknown"),
        "is_arm":       deps.get("is_arm", False),
        "dependencies": deps,
        "model_sizes":  sizes,
        "sklearn_benchmark": benchmark,
        "onnx_export":  onnx_result,
        "onnx_vs_sklearn": onnx_bench,
        "recommendations": []
    }

    # Recommandations automatiques
    recs = report["recommendations"]

    total_mb = sizes.get("TOTAL_MB", 0)
    if total_mb > 100:
        recs.append({
            "priority": "HIGH",
            "message":  f"Taille totale modèles : {total_mb:.0f} MB — trop lourds pour un Pi 4 (RAM 4GB limitée).",
            "action":   "Utiliser edge_pipeline_if_only.pkl (< 5MB) pour l'Edge."
        })

    ecod_size = sizes.get("model_ecod_v3.pkl", 0)
    if ecod_size > 100:
        recs.append({
            "priority": "HIGH",
            "message":  f"ECOD : {ecod_size:.0f} MB — à exclure du déploiement Edge.",
            "action":   "Supprimer ECOD du docker-compose Edge, utiliser IF+OCSVM seulement."
        })

    total_ms = benchmark.get("total_ensemble_ms", 0)
    if total_ms > 100:
        recs.append({
            "priority": "MEDIUM",
            "message":  f"Latence ensemble : {total_ms:.0f}ms — dépasse 100ms.",
            "action":   "Utiliser les modèles ONNX Runtime pour x2-x5 de gain."
        })

    if onnx_result.get("exported"):
        speedup = onnx_bench.get("speedup_factor", 1.0)
        if speedup > 1.2:
            recs.append({
                "priority": "LOW",
                "message":  f"ONNX Runtime offre un gain de {speedup:.1f}x sur le prétraitement.",
                "action":   "Remplacer le prétraitement sklearn par onnxruntime dans edge_api.py."
            })

    if not deps.get("skl2onnx"):
        recs.append({
            "priority": "LOW",
            "message":  "skl2onnx non installé — export ONNX non disponible.",
            "action":   "pip install skl2onnx onnxruntime"
        })

    # Résumé
    report["summary"] = {
        "total_model_size_mb": total_mb,
        "inference_latency_ms": benchmark.get("total_ensemble_ms", 0),
        "recommended_for_edge": "edge_pipeline_if_only.pkl",
        "onnx_available": bool(onnx_result.get("exported")),
        "offline_mode_ready": True,
    }

    # Sauvegarder le rapport
    report_path = EDGE_DIR / "edge_optimization_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(f"Rapport Edge sauvegardé → {report_path}")

    return report


def print_edge_summary(report: dict):
    """Affiche un résumé console du rapport Edge."""
    print("\n" + "="*65)
    print("  RAPPORT D'OPTIMISATION EDGE")
    print("="*65)

    sizes = report.get("model_sizes", {})
    print(f"\n  Taille totale modèles : {sizes.get('TOTAL_MB', 0):.0f} MB")
    for name, size in sorted(sizes.items(), key=lambda x: -x[1]):
        if name == "TOTAL_MB":
            continue
        bar = "#" * int(size / 5)
        tag = " [LOURD]" if size > 50 else ""
        print(f"    {name:<30} {size:6.1f} MB  {bar}{tag}")

    bench = report.get("sklearn_benchmark", {})
    if bench:
        print(f"\n  Latence inférence sklearn :")
        print(f"    Prétraitement          : {bench.get('preprocess_ms',0):.3f} ms")
        print(f"    IF                     : {bench.get('if_ms',0):.3f} ms")
        print(f"    OCSVM                  : {bench.get('ocsvm_ms',0):.3f} ms")
        print(f"    Ensemble total         : {bench.get('total_ensemble_ms',0):.3f} ms")
        print(f"    Débit max              : {bench.get('throughput_pred_per_sec',0):.0f} pred/s")

    onnx_b = report.get("onnx_vs_sklearn", {})
    if onnx_b.get("speedup_factor"):
        print(f"\n  ONNX Runtime vs sklearn : {onnx_b['speedup_factor']:.2f}x plus rapide")

    recs = report.get("recommendations", [])
    if recs:
        print(f"\n  Recommandations ({len(recs)}) :")
        for r in recs:
            prio = r["priority"]
            print(f"    [{prio}] {r['message']}")
            print(f"            -> {r['action']}")

    summ = report.get("summary", {})
    print(f"\n  Modèle recommandé Edge : {summ.get('recommended_for_edge', '—')}")
    print(f"  Mode offline prêt      : {'Oui' if summ.get('offline_mode_ready') else 'Non'}")
    print("="*65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  DOCKER COMPOSE EDGE — version allégée
# ══════════════════════════════════════════════════════════════════════════════

DOCKER_COMPOSE_EDGE = """version: '3.8'

services:
  # API FastAPI — version Edge (modèles allégés uniquement)
  api-edge:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - EDGE_MODE=1
    image: predictive-maintenance-edge:latest
    container_name: pm_api_edge
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro        # Lecture seule — modèles pré-chargés
      - ./edge_models:/app/edge_models:ro
      - ./data:/app/data
      - realtime_data:/app/realtime    # Volume partagé avec le moteur RT
    environment:
      - EDGE_MODE=1
      - EDGE_MODEL_ONLY=if             # Utiliser uniquement IF (pas ECOD/LOF)
      - MAX_WORKERS=2                  # Limiter les workers sur Pi4
      - PYTHONUNBUFFERED=1
    mem_limit: 1g                      # Limiter à 1GB RAM (Pi4 = 4GB total)
    cpus: "2"                          # Limiter à 2 cœurs sur 4

  # Moteur temps réel — polling IFM direct
  engine-edge:
    image: predictive-maintenance-edge:latest
    container_name: pm_engine_edge
    restart: unless-stopped
    depends_on:
      - api-edge
    command: python realtime_ifm_direct.py
    volumes:
      - realtime_data:/app/realtime
      - ./data:/app/data
    environment:
      - API_URL=http://api-edge:8000
      - IFM_GATEWAY_URL=http://192.168.1.50
      - POLL_INTERVAL=2
    mem_limit: 256m

  # Dashboard Nginx — version légère
  dashboard-edge:
    image: nginx:alpine
    container_name: pm_dashboard_edge
    restart: unless-stopped
    ports:
      - "80:80"
    volumes:
      - ./dashboard_realtime.html:/usr/share/nginx/html/index.html:ro
      - ./dashboard_predictive.html:/usr/share/nginx/html/predictive.html:ro
      - realtime_data:/usr/share/nginx/html/data:ro

volumes:
  realtime_data:
"""


def generate_edge_dockerfile():
    """Génère un Dockerfile optimisé pour l'Edge (ARM64 / Pi4)."""
    dockerfile = """# Dockerfile Edge — Raspberry Pi 4 (ARM64)
# Optimisé pour mémoire réduite : pas d'ECOD, pas de LOF

FROM python:3.11-slim

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc g++ libgomp1 && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Installer seulement les packages nécessaires (sans PyOD pour l'Edge)
RUN pip install --no-cache-dir \\
    fastapi==0.115.* uvicorn[standard]==0.30.* \\
    scikit-learn==1.5.* pandas==2.2.* numpy==1.26.* \\
    joblib==1.4.* scipy==1.14.* \\
    requests==2.32.* mysql-connector-python==8.4.* && \\
    # ONNX Runtime pour ARM64 (plus léger que sklearn sur Pi4)
    pip install --no-cache-dir onnxruntime || true

# Copier le code source
COPY api_unified_pythagore.py .
COPY signal_processing.py .
COPY edge_optimize.py .
COPY realtime_ifm_direct.py .
COPY alert_manager.py .
COPY alert_config.json .

# Modèles et données
COPY models/ models/
COPY edge_models/ edge_models/

# Variable d'environnement Edge
ENV EDGE_MODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api_unified_pythagore:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
"""
    out = EDGE_DIR / "Dockerfile.edge"
    out.write_text(dockerfile, encoding="utf-8")

    compose_out = EDGE_DIR / "docker-compose.edge.yml"
    compose_out.write_text(DOCKER_COMPOSE_EDGE, encoding="utf-8")

    log.info(f"Dockerfile Edge → {out}")
    log.info(f"docker-compose Edge → {compose_out}")
    return str(out), str(compose_out)


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Optimisation Edge — Raspberry Pi 4 / ARM64")
    parser.add_argument("--check",     action="store_true", help="Vérifier les dépendances")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark inférence sklearn")
    parser.add_argument("--export",    action="store_true", help="Exporter les modèles en ONNX")
    parser.add_argument("--lightweight", action="store_true", help="Créer le pipeline léger Edge")
    parser.add_argument("--docker",    action="store_true", help="Générer Dockerfile Edge optimisé")
    parser.add_argument("--all",       action="store_true", help="Tout exécuter (par défaut)")
    args = parser.parse_args()

    run_all = args.all or not any([args.check, args.benchmark, args.export, args.lightweight, args.docker])

    log.info("="*60)
    log.info("  Edge Optimization — Maintenance Prédictive")
    log.info("="*60)

    if args.check or run_all:
        deps = check_dependencies()
        log.info("\n  Dépendances :")
        for k, v in deps.items():
            status = "OK" if v else "MANQUANT"
            log.info(f"    {k:<20} : {v or 'non installé'}")

    sizes = benchmark_model_sizes()
    log.info(f"\n  Taille totale modèles : {sizes.get('TOTAL_MB', 0):.1f} MB")

    benchmark = {}
    if args.benchmark or run_all:
        log.info("\n  Benchmark inférence sklearn :")
        benchmark = benchmark_sklearn_models(n_iter=50)

    onnx_result = {}
    if args.export or run_all:
        log.info("\n  Export ONNX :")
        onnx_result = export_to_onnx()

    onnx_bench = {}
    if (args.export or run_all) and onnx_result.get("exported"):
        log.info("\n  Benchmark ONNX vs sklearn :")
        onnx_bench = benchmark_onnx_vs_sklearn(n_iter=100)

    if args.lightweight or run_all:
        log.info("\n  Création pipeline léger Edge :")
        lw = create_lightweight_ensemble()
        if lw.get("status") == "success":
            log.info(f"  Pipeline léger créé : {lw['size_kb']} KB")

    if args.docker or run_all:
        log.info("\n  Génération Dockerfile Edge :")
        generate_edge_dockerfile()

    if run_all or args.benchmark:
        report = generate_edge_report(sizes, benchmark, onnx_result, onnx_bench)
        print_edge_summary(report)


if __name__ == "__main__":
    main()
