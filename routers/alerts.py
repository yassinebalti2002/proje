"""
routers/alerts.py
==================
Consultation de l'historique et des statistiques des alertes externes
(email / webhook / SMS) envoyées par alert_manager.py.
"""

from fastapi import APIRouter, Request, Depends

import core
from rate_limiter import make_rate_limiter

router = APIRouter(tags=["Alertes"])


# ── Historique alertes externes ────────────────────────────────────────
@router.get("/v1/alerts", summary="Historique des alertes externes envoyées")
def get_alerts_history(request: Request, limit: int = 50, _rl=Depends(make_rate_limiter(30))):
    """
    Retourne les dernières alertes envoyées via email/webhook/SMS.
    Inclut le statut de livraison par canal (booléens uniquement, jamais
    d'adresse/URL/identifiant — public, comme les autres endpoints de
    monitoring en lecture seule).
    Nécessite alert_config.json configuré.
    """
    if not core.ALERTS_ENABLED or core._alert_manager is None:
        return {
            "enabled": False,
            "message": "AlertManager non disponible. Vérifier alert_config.json",
            "alerts": []
        }
    return {
        "enabled": True,
        "stats":   core._alert_manager.get_stats(),
        "alerts":  core._alert_manager.get_history(limit=limit)
    }


@router.get("/v1/alerts/stats", summary="Statistiques du gestionnaire d'alertes")
def get_alerts_stats(request: Request, _rl=Depends(make_rate_limiter(30))):
    """Statistiques globales : total envoyées, par niveau, cooldowns actifs."""
    if not core.ALERTS_ENABLED or core._alert_manager is None:
        return {"enabled": False, "channels": "aucun", "total_alerts": 0}
    return {"enabled": True, **core._alert_manager.get_stats()}
