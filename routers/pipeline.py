"""
routers/pipeline.py
====================
Pipeline de ré-entraînement : upload d'un dump SQL → parsing → entraînement
→ suivi de progression. Chaque run est isolé dans models/runs/, jamais dans
models/ (production) — voir _run_pipeline_bg.
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse

import core
from auth import require_api_key, require_admin_key
from rate_limiter import make_rate_limiter

log = logging.getLogger(__name__)
router = APIRouter(tags=["Pipeline"])

# Stockage en mémoire des jobs pipeline (clé = job_id)
_pipeline_jobs: dict = {}

# Historique persisté des jobs terminés (survit au redémarrage, contrairement
# à _pipeline_jobs ci-dessus qui est vidé à chaque relance de l'API) --
# alimente la page d'historique des tâches du dashboard.
_PIPELINE_HISTORY_PATH = Path(__file__).parent.parent / "pipeline_jobs_history.json"
_MAX_PIPELINE_HISTORY = 300

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mGKHF]|\x1b\(B')


def _persist_job_history(job: dict):
    """Ajoute le job terminé (done/error) à l'historique persisté sur disque."""
    try:
        history = []
        if _PIPELINE_HISTORY_PATH.exists():
            history = json.loads(_PIPELINE_HISTORY_PATH.read_text(encoding="utf-8"))
        history.append({
            "job_id":      job["job_id"],
            "status":      job["status"],
            "filename":    job.get("filename"),
            "elapsed":     job.get("elapsed"),
            "created_at":  job.get("created_at"),
            "finished_at": datetime.now().isoformat(),
            "results":     job.get("results"),
        })
        history = history[-_MAX_PIPELINE_HISTORY:]
        _PIPELINE_HISTORY_PATH.write_text(
            json.dumps(history, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    except Exception as e:
        log.warning(f"Persistence historique pipeline échouée : {e}")


def _fmt_elapsed(start_ts: float) -> str:
    s = int(time.time() - start_ts)
    return f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"


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
    run_dir = Path("models/runs") / f"{datetime.now():%Y%m%d_%H%M%S}_{job_id}"
    job["run_dir"] = str(run_dir)

    try:
        start = time.time()
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
            [sys.executable, "-u", "generate_dataset_from_sql.py",
             "--sql", sql_path, "--out", csv_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=_env,
            cwd=str(Path(__file__).parent.parent)
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
            [sys.executable, "-u", "train_model_v3_unsupervised.py",
             "--csv", csv_path, "--out-dir", str(run_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=_env,
            cwd=str(Path(__file__).parent.parent)
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
            if core.df_results is not None and not core.df_results.empty:
                top = core.df_results.nlargest(8, "anomaly_score")
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
        _persist_job_history(job)

    except Exception as exc:
        job["status"] = "error"
        _add_log(job, f"❌ Erreur fatale : {exc}", "e")
        log.error(f"Pipeline {job_id} failed: {exc}")
        _persist_job_history(job)
    finally:
        # Nettoyage fichiers temporaires
        for p in [sql_path, locals().get('csv_path')]:
            try:
                if p and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass


# ── Endpoint : page HTML pipeline ─────────────────────────────────────────
@router.get("/pipeline", include_in_schema=False)
def get_pipeline_page():
    """Sert la page web d'upload SQL."""
    html_path = Path(__file__).parent.parent / "pipeline_upload.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return HTMLResponse("<h1>pipeline_upload.html introuvable</h1>", status_code=404)


# ── Endpoint : upload SQL + lancement pipeline ────────────────────────────
@router.post(
    "/v1/pipeline/upload",
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

    job_id = str(uuid.uuid4())[:12]

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
        "created_at": datetime.now().isoformat(),
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
@router.get(
    "/v1/pipeline/status/{job_id}",
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
@router.get("/v1/pipeline/jobs", summary="Liste des pipelines récents")
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


# ── Endpoint : historique complet persisté ─────────────────────────────────
@router.get(
    "/v1/pipeline/jobs/history",
    summary="Historique complet des runs pipeline (persisté sur disque)",
    description=(
        "Contrairement à /v1/pipeline/jobs (10 derniers, en mémoire, perdus au "
        "redémarrage), cet endpoint lit pipeline_jobs_history.json : tous les "
        "runs terminés (succès ou erreur) depuis le début, jusqu'à 300 entrées."
    )
)
def pipeline_jobs_history(request: Request, limit: int = 100, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(30))):
    if not _PIPELINE_HISTORY_PATH.exists():
        return {"total": 0, "jobs": []}
    try:
        history = json.loads(_PIPELINE_HISTORY_PATH.read_text(encoding="utf-8"))
        return {"total": len(history), "jobs": list(reversed(history[-limit:]))}
    except Exception as e:
        log.warning(f"/v1/pipeline/jobs/history erreur lecture : {e}")
        return {"total": 0, "jobs": [], "error": str(e)}
