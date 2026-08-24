"""
test_ml_functions.py
====================
Tests unitaires des fonctions ML de l'API (sans démarrer le serveur).

Couvre :
  - Fonctions statistiques : safe_mean, safe_std, safe_rms, safe_trend, norm01
  - Vibration totale 3D : vib_total_pythagorean
  - Extraction de features : extract_features (sorties et bornes)
  - Auth : require_api_key (401 sans clé, 200 avec clé valide)

Usage :
    pytest tests/test_ml_functions.py -v
"""

import os
import math
import sys
import pytest
import numpy as np

# ── Chargement du module sans démarrer le serveur ─────────────────────────────
os.environ.setdefault("API_KEYS", "test-key-unit")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import core as api


# ══════════════════════════════════════════════════════════════════════════════
#  Fonctions statistiques de base
# ══════════════════════════════════════════════════════════════════════════════

class TestSafeMean:
    def test_liste_normale(self):
        assert api.safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_liste_vide_retourne_nan(self):
        assert math.isnan(api.safe_mean([]))

    def test_un_seul_element(self):
        assert api.safe_mean([42.0]) == pytest.approx(42.0)


class TestSafeStd:
    def test_liste_normale(self):
        result = api.safe_std([1.0, 2.0, 3.0, 4.0])
        assert result > 0.0

    def test_un_element_retourne_zero(self):
        assert api.safe_std([5.0]) == 0.0

    def test_liste_vide_retourne_zero(self):
        assert api.safe_std([]) == 0.0

    def test_valeurs_identiques(self):
        assert api.safe_std([7.0, 7.0, 7.0]) == pytest.approx(0.0)


class TestSafeRms:
    def test_valeurs_positives(self):
        # RMS([3, 4]) = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.535
        result = api.safe_rms([3.0, 4.0])
        assert result == pytest.approx(math.sqrt(12.5), rel=1e-5)

    def test_liste_vide_retourne_nan(self):
        assert math.isnan(api.safe_rms([]))

    def test_valeur_constante(self):
        assert api.safe_rms([5.0, 5.0, 5.0]) == pytest.approx(5.0)


class TestSafeTrend:
    def test_tendance_croissante(self):
        # Série croissante → pente positive
        result = api.safe_trend([1.0, 2.0, 3.0, 4.0])
        assert result > 0.0

    def test_tendance_decroissante(self):
        result = api.safe_trend([4.0, 3.0, 2.0, 1.0])
        assert result < 0.0

    def test_serie_constante(self):
        result = api.safe_trend([5.0, 5.0, 5.0, 5.0])
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_moins_de_trois_points(self):
        assert api.safe_trend([1.0, 2.0]) == 0.0
        assert api.safe_trend([]) == 0.0


class TestNorm01:
    def test_valeur_dans_plage(self):
        # norm01(50, 0, 100) = 0.5
        result = api.norm01(50.0, 0.0, 100.0)
        assert result == pytest.approx(0.5, rel=1e-5)

    def test_valeur_min_retourne_zero(self):
        assert api.norm01(0.0, 0.0, 100.0) == pytest.approx(0.0, abs=1e-5)

    def test_valeur_max_retourne_un(self):
        assert api.norm01(100.0, 0.0, 100.0) == pytest.approx(1.0, rel=1e-5)

    def test_valeur_inferieure_clampee_a_zero(self):
        assert api.norm01(-10.0, 0.0, 100.0) == 0.0

    def test_valeur_superieure_clampee_a_un(self):
        assert api.norm01(200.0, 0.0, 100.0) == 1.0

    def test_nan_retourne_05(self):
        assert api.norm01(float("nan"), 0.0, 100.0) == 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  Vibration totale 3D (Pythagore)
# ══════════════════════════════════════════════════════════════════════════════

class TestVibTotalPythagorean:
    def test_triplet_classique(self):
        # sqrt(3² + 4² + 0²) = 5
        result = api.vib_total_pythagorean(3.0, 4.0, 0.0)
        assert result == pytest.approx(5.0)

    def test_tous_nuls(self):
        assert api.vib_total_pythagorean(0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_axes_egaux(self):
        # sqrt(3 * v²) = v * sqrt(3)
        v = 500.0
        expected = v * math.sqrt(3)
        assert api.vib_total_pythagorean(v, v, v) == pytest.approx(expected, rel=1e-6)

    def test_resultat_toujours_positif(self):
        assert api.vib_total_pythagorean(100.0, 200.0, 300.0) > 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Extraction de features
# ══════════════════════════════════════════════════════════════════════════════

def _make_history(n=10, temp=35.0, vib=300.0):
    """Génère une liste de MeasurePoint avec des valeurs uniformes."""
    return [
        api.MeasurePoint(
            temperature=temp,
            vibration_x=vib,
            vibration_y=vib + 5,
            vibration_z=vib + 10,
        )
        for _ in range(n)
    ]


class TestExtractFeatures:
    def test_retourne_toutes_les_cles_attendues(self):
        history = _make_history()
        feat = api.extract_features(history)
        required_keys = [
            "temp_mean", "temp_std", "temp_trend", "temp_cur",
            "vib_z_mean", "vib_z_std", "vib_z_rms_w", "vib_z_kurt",
            "vib_z_crest", "vib_z_cur",
            "vib_x_mean", "vib_y_mean",
            "vib_total", "health_score",
            "delta_vib", "delta_temp", "vib_entropy", "fft_ratio",
        ]
        for k in required_keys:
            assert k in feat, f"Clé manquante : {k}"

    def test_temperature_coherente(self):
        history = _make_history(temp=40.0)
        feat = api.extract_features(history)
        assert feat["temp_mean"] == pytest.approx(40.0, abs=0.1)
        assert feat["temp_cur"] == pytest.approx(40.0, abs=0.1)

    def test_vib_total_positif(self):
        history = _make_history(vib=300.0)
        feat = api.extract_features(history)
        assert feat["vib_total"] > 0.0

    def test_health_score_dans_bornes(self):
        history = _make_history(temp=35.0, vib=300.0)
        feat = api.extract_features(history)
        assert 0.0 <= feat["health_score"] <= 100.0

    def test_serie_constante_std_nulle(self):
        history = _make_history(n=10, temp=32.0, vib=270.0)
        feat = api.extract_features(history)
        # Signal constant → std ≈ 0, trend ≈ 0
        assert feat["temp_std"] == pytest.approx(0.0, abs=0.01)
        assert feat["temp_trend"] == pytest.approx(0.0, abs=0.01)

    def test_une_seule_mesure(self):
        history = _make_history(n=1)
        feat = api.extract_features(history)
        # Ne doit pas lever d'exception
        assert "temp_mean" in feat

    def test_temperature_elevee_degrade_health(self):
        history_normal = _make_history(temp=30.0, vib=200.0)
        history_hot    = _make_history(temp=65.0, vib=200.0)
        feat_normal = api.extract_features(history_normal)
        feat_hot    = api.extract_features(history_hot)
        assert feat_hot["health_score"] < feat_normal["health_score"]

    def test_vibration_elevee_degrade_health(self):
        history_calm  = _make_history(temp=35.0, vib=100.0)
        history_loud  = _make_history(temp=35.0, vib=1200.0)
        feat_calm = api.extract_features(history_calm)
        feat_loud = api.extract_features(history_loud)
        assert feat_loud["health_score"] < feat_calm["health_score"]


# ══════════════════════════════════════════════════════════════════════════════
#  RUL — non-régression du bug corrigé cette session : un alert_level
#  escaladé via predict_risk="CRITIQUE" avec une dégradation physique faible
#  pouvait produire un rul_hours hors de la plage RUL_TABLE de son propre
#  alert_level (observé en réel : alert_level="URGENT" mais rul_hours=307h,
#  hors de [72,168]h — texte "RUL 3-7 jours" contredit par 307h affichées).
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeRul:
    @pytest.mark.parametrize("risk_level", ["CRITIQUE", "ÉLEVÉ", "MODÉRÉ", "FAIBLE", ""])
    def test_rul_hours_toujours_dans_la_plage_de_son_alert_level(self, risk_level):
        # Dégradation physique modérée (santé < 85 pour ne pas déclencher le
        # court-circuit _force_ok qui bypasse tout le calcul de ratio)
        history = _make_history(n=10, temp=41.0, vib=250.0)
        feat = api.extract_features(history)
        assert feat["health_score"] < 85, "précondition : health_score doit être < 85"

        predict_result = {"risk_level": risk_level, "anomaly_score": 0.6}
        result = api.compute_rul(
            history, feat, f"test_rul_regress_{risk_level or 'fallback'}", predict_result
        )

        lo, hi = api.RUL_TABLE[result["alert_level"]]
        assert lo <= result["rul_hours"] <= hi, (
            f"risk_level={risk_level!r} -> alert_level={result['alert_level']} "
            f"mais rul_hours={result['rul_hours']} hors de [{lo},{hi}]"
        )

    def test_critique_ml_avec_faible_degradation_ne_force_pas_urgent(self):
        """
        Non-régression : reproduit le cas réel observé en production (replay)
        où predict_risk="CRITIQUE" (score ML saturé) + dégradation physique
        faible (~19%) forçait quand même alert_level="URGENT" pour la quasi
        totalité du parc simultanément -- dashboard à 90% "en anomalie" et
        flood d'alertes email. Doit maintenant retomber en "ATTENTION".
        """
        feat = {
            "health_score": 67.9, "temp_mean": 40.87, "vib_total": 861.98,
            "vib_z_kurt": 2.21, "vib_z_crest": 1.15,
        }
        history = _make_history(n=10, temp=40.9, vib=250.0)
        predict_result = {"risk_level": "CRITIQUE", "anomaly_score": 0.9959}

        result = api.compute_rul(history, feat, "test_rul_critique_lowdeg", predict_result)

        assert result["alert_level"] == "ATTENTION", (
            f"alert_level={result['alert_level']} -- une dégradation physique "
            f"faible ne devrait pas forcer URGENT même si le ML dit CRITIQUE"
        )
        lo, hi = api.RUL_TABLE["ATTENTION"]
        assert lo <= result["rul_hours"] <= hi

    def test_health_score_eleve_force_ok(self):
        # Capteur clairement sain : health_score >= 85 doit toujours forcer
        # alert_level="OK" avec un plancher de 336h, quel que soit risk_level.
        history = _make_history(n=10, temp=30.0, vib=50.0)
        feat = api.extract_features(history)
        assert feat["health_score"] >= 85, "précondition : capteur sain"

        result = api.compute_rul(
            history, feat, "test_rul_force_ok", {"risk_level": "CRITIQUE", "anomaly_score": 0.9}
        )
        assert result["alert_level"] == "OK"
        assert result["rul_hours"] >= 336.0


# ══════════════════════════════════════════════════════════════════════════════
#  Authentification
# ══════════════════════════════════════════════════════════════════════════════

class TestAuth:
    def test_comparaison_temps_constant_cle_valide(self):
        from auth import _constant_time_compare
        assert _constant_time_compare("abc", "abc") is True

    def test_comparaison_temps_constant_cle_invalide(self):
        from auth import _constant_time_compare
        assert _constant_time_compare("abc", "xyz") is False

    def test_cles_chargees_depuis_env(self):
        import importlib
        os.environ["API_KEYS"] = "cle1,cle2,cle3"
        import auth
        importlib.reload(auth)
        assert "cle1" in auth.VALID_API_KEYS
        assert "cle2" in auth.VALID_API_KEYS
        assert "cle3" in auth.VALID_API_KEYS
        # Restaurer
        os.environ["API_KEYS"] = "test-key-unit"
        importlib.reload(auth)

    def test_cles_avec_espaces_ignorees(self):
        import importlib
        os.environ["API_KEYS"] = " cle1 , , cle2 "
        import auth
        importlib.reload(auth)
        assert "cle1" in auth.VALID_API_KEYS
        assert "" not in auth.VALID_API_KEYS
        os.environ["API_KEYS"] = "test-key-unit"
        importlib.reload(auth)
