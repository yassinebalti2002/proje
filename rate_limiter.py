"""
rate_limiter.py
===============
Rate limiter in-memory basé sur une fenêtre glissante.
Implémenté comme dépendance FastAPI native — pas de middleware externe.

Usage :
    from rate_limiter import make_rate_limiter

    @app.post("/v1/predict")
    def predict(..., _rl=Depends(make_rate_limiter(30))):
        ...

Thread-safe via threading.Lock.
"""

import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request, status

_store: dict[str, list[float]] = defaultdict(list)
_lock = Lock()

# Balayage global périodique : purge les clés (IP, path) totalement inactives.
# Sans ça, un client non authentifié qui varie l'URL (ex: /v1/history/{id}
# avec un id différent à chaque appel) fait grossir _store indéfiniment,
# même en restant sous la limite — fuite mémoire exploitable sans auth.
_SWEEP_INTERVAL = 300.0   # secondes
_STALE_AFTER    = 3600.0  # une clé inactive depuis 1h est purgée (> toute fenêtre utilisée dans l'app)
_last_sweep = 0.0


def _sweep_locked(now: float) -> None:
    """Doit être appelé avec _lock déjà acquis."""
    global _last_sweep
    if now - _last_sweep < _SWEEP_INTERVAL:
        return
    _last_sweep = now
    dead_keys = [
        k for k, ts in _store.items()
        if not ts or ts[-1] <= now - _STALE_AFTER
    ]
    for k in dead_keys:
        del _store[k]


def make_rate_limiter(max_calls: int, window_seconds: int = 60):
    """
    Crée une dépendance FastAPI qui bloque avec 429 après max_calls
    requêtes par IP dans la fenêtre glissante de window_seconds secondes.

    La clé inclut le sensor_id (extrait du corps JSON pour les endpoints POST
    type /v1/predict) quand il est présent. Sans ça, tous les capteurs d'un
    même moteur temps réel (même IP source) partagent le même quota puisque
    /v1/predict et /v1/predict-rul sont des chemins FIXES -- le sensor_id est
    dans le corps, pas dans l'URL. Avec ~19-20 capteurs actifs, ce quota
    partagé se vidait en quelques secondes et bloquait le moteur en usage
    normal (pas seulement en cas d'abus). Les endpoints GET paramétrés par
    l'URL (ex: /v1/alert-level/{sensor_id}) n'étaient déjà pas concernés :
    request.url.path contient déjà le sensor_id réel pour ceux-là.
    """
    async def _limiter(request: Request) -> None:
        ip = (request.client.host if request.client else "unknown")
        sensor_suffix = ""
        if request.method == "POST":
            try:
                body = await request.json()
                sid = body.get("sensor_id") if isinstance(body, dict) else None
                if sid:
                    sensor_suffix = f":{sid}"
            except Exception:
                pass   # corps non-JSON (ex: upload multipart) -- comportement inchangé
        key = f"{ip}:{request.url.path}{sensor_suffix}"
        now = time.monotonic()
        cutoff = now - window_seconds

        with _lock:
            _sweep_locked(now)
            timestamps = _store[key]
            # Supprimer les appels hors fenêtre
            timestamps[:] = [t for t in timestamps if t > cutoff]
            if len(timestamps) >= max_calls:
                oldest = timestamps[0]
                retry_after = max(1, int(window_seconds - (now - oldest)) + 1)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Trop de requetes. Limite : {max_calls} appels/{window_seconds}s par IP. "
                        f"Reessayez dans {retry_after}s."
                    ),
                    headers={"Retry-After": str(retry_after)},
                )
            timestamps.append(now)

    return _limiter
