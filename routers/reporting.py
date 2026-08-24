"""
routers/reporting.py
=====================
Génération de rapports de maintenance HTML/JSON (reporting_module.py).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse

import core
from auth import require_api_key
from rate_limiter import make_rate_limiter

log = logging.getLogger(__name__)
router = APIRouter(tags=["Reporting"])


@router.get(
    "/v1/report",
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
    if not core.REPORTING_OK:
        raise HTTPException(
            status_code=503,
            detail="Module reporting_module non disponible."
        )
    if type not in ("daily", "weekly", "monthly", "full"):
        raise HTTPException(status_code=400, detail="type doit être : daily | weekly | monthly | full")

    from reporting_module import generate_html_report, generate_json_report, save_report
    try:
        if format == "html":
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
