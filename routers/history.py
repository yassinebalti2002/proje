"""
routers/history.py
====================
Vue d'ensemble historique du dashboard :
- GET /v1/tasks-history : journal unifié de toutes les tâches exécutées par
  le système (runs pipeline ML, alertes envoyées, rapports générés), triées
  du plus récent au plus ancien -- réponse à "qu'est-ce qui s'est passé dans
  le dashboard, et quand ?".
- Pages HTML kpi-history.html et tasks-history.html servies directement par
  l'API (même principe que routers/pages.py).

Le KPI history proprement dit (GET /v1/kpi-history) vit dans routers/data.py,
à côté des autres endpoints de consultation de données -- il n'est pas
dupliqué ici.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse

import core
from auth import require_api_key
from rate_limiter import make_rate_limiter

log = logging.getLogger(__name__)
router = APIRouter(tags=["Historique"])

_PROJECT_DIR = Path(__file__).parent.parent
_PIPELINE_HISTORY_PATH = _PROJECT_DIR / "pipeline_jobs_history.json"
_REPORTS_DIR = _PROJECT_DIR / "reports"


# ── Pages HTML ──────────────────────────────────────────────────────────────
@router.get("/kpi-history.html", include_in_schema=False)
def get_kpi_history_page():
    html_path = _PROJECT_DIR / "kpi_history.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return HTMLResponse("<h1>kpi_history.html introuvable</h1>", status_code=404)


@router.get("/tasks-history.html", include_in_schema=False)
def get_tasks_history_page():
    html_path = _PROJECT_DIR / "tasks_history.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return HTMLResponse("<h1>tasks_history.html introuvable</h1>", status_code=404)


# ── Journal unifié des tâches ────────────────────────────────────────────────
def _pipeline_events() -> list:
    events = []
    if not _PIPELINE_HISTORY_PATH.exists():
        return events
    try:
        for j in json.loads(_PIPELINE_HISTORY_PATH.read_text(encoding="utf-8")):
            results = j.get("results") or {}
            detail = f"durée {j.get('elapsed', '?')}"
            if "auc" in results:
                detail = f"AUC={results['auc']:.3f} · F1={results.get('f1', 0):.3f} · durée {j.get('elapsed', '?')}"
            events.append({
                "type":      "pipeline",
                "icon":      "🧠",
                "timestamp": j.get("finished_at") or j.get("created_at"),
                "title":     f"Entraînement modèle — {j.get('filename', '?')}",
                "status":    j.get("status", "?"),
                "detail":    detail,
            })
    except Exception as e:
        log.warning(f"Lecture historique pipeline échouée : {e}")
    return events


def _alert_events() -> list:
    events = []
    if not core.ALERTS_ENABLED or core._alert_manager is None:
        return events
    try:
        for a in core._alert_manager.get_history(limit=300):
            channels_ok = [c for c, ok in (a.get("delivery") or {}).items() if ok]
            events.append({
                "type":      "alert",
                "icon":      "🔔",
                "timestamp": a.get("timestamp"),
                "title":     f"Alerte {a.get('risk_level', '?')} — capteur {a.get('sensor_id', '?')}",
                "status":    "envoyée" if channels_ok else "échec",
                "detail":    f"Canaux : {', '.join(channels_ok) or 'aucun'}",
            })
    except Exception as e:
        log.warning(f"Lecture historique alertes échouée : {e}")
    return events


def _report_events() -> list:
    events = []
    if not _REPORTS_DIR.exists():
        return events
    try:
        for f in sorted(_REPORTS_DIR.glob("rapport_*.html")):
            from datetime import datetime
            events.append({
                "type":      "report",
                "icon":      "📄",
                "timestamp": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "title":     f"Rapport généré — {f.name}",
                "status":    "généré",
                "detail":    f"{f.stat().st_size / 1024:.1f} Ko",
            })
    except Exception as e:
        log.warning(f"Lecture dossier reports/ échouée : {e}")
    return events


@router.get(
    "/v1/tasks-history",
    summary="Journal unifié des tâches du dashboard (pipeline, alertes, rapports)",
    description=(
        "Agrège en une seule chronologie, du plus récent au plus ancien :\n\n"
        "- **pipeline** : runs d'entraînement ML (upload SQL → parsing → entraînement)\n"
        "- **alert** : alertes email/webhook/SMS envoyées\n"
        "- **report** : rapports de maintenance générés (GET /v1/report?format=html)\n\n"
        "Filtrer avec `type=pipeline|alert|report` pour n'en garder qu'une catégorie."
    )
)
def get_tasks_history(
    request: Request,
    type: str = None,
    limit: int = 100,
    _key: str = Depends(require_api_key),
    _rl=Depends(make_rate_limiter(30)),
):
    events = _pipeline_events() + _alert_events() + _report_events()

    if type:
        events = [e for e in events if e["type"] == type]

    events.sort(key=lambda e: e.get("timestamp") or "", reverse=True)
    events = events[:limit]

    return {
        "total":  len(events),
        "counts": {
            "pipeline": sum(1 for e in events if e["type"] == "pipeline"),
            "alert":    sum(1 for e in events if e["type"] == "alert"),
            "report":   sum(1 for e in events if e["type"] == "report"),
        },
        "events": events,
    }
