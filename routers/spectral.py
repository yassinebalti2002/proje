"""
routers/spectral.py
====================
Analyse spectrale FFT et détection de défauts de roulements.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Request, Depends, HTTPException

import core
from core import PredictRequest
from auth import require_api_key
from rate_limiter import make_rate_limiter

log = logging.getLogger(__name__)
router = APIRouter(tags=["IA / Prédiction"])


# ══════════════════════════════════════════════════════════════════════════
#  POST /v1/spectral-analysis — Analyse spectrale FFT + défauts roulements
# ══════════════════════════════════════════════════════════════════════════
@router.post(
    "/v1/spectral-analysis",
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
    if not core.SIGNAL_PROCESSING_OK:
        raise HTTPException(
            status_code=503,
            detail="Module signal_processing non disponible. Vérifier l'installation de scipy."
        )
    from signal_processing import extract_spectral_features, full_signal_pipeline

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


# ══════════════════════════════════════════════════════════════════════════
#  GET /v1/spectral/{sensor_id} — Analyse spectrale publique (dashboard)
# ══════════════════════════════════════════════════════════════════════════
@router.get(
    "/v1/spectral/{sensor_id}",
    summary="Analyse spectrale publique à partir du buffer serveur",
    description=(
        "Version publique (lecture seule, sans clé) de l'analyse spectrale — "
        "utilise les dernières valeurs de vibration_z déjà reçues via /v1/predict "
        "pour ce capteur (pas de recalcul côté client). Pensée pour le dashboard."
    )
)
def spectral_public(request: Request, sensor_id: str, rpm: float = 1450.0, fs: float = 100.0,
                     _rl=Depends(make_rate_limiter(30))):
    if not core.SIGNAL_PROCESSING_OK:
        return {"available": False, "reason": "Module signal_processing non disponible."}
    from signal_processing import full_signal_pipeline
    buf = core._raw_vib_buffers.get(sensor_id)
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
