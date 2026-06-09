"""
reporting_module.py
====================
Système de reporting pour l'analyse de maintenance.

Fonctionnalités :
    - Rapport HTML complet (résumé exécutif, KPIs, alertes, tableau de bord)
    - Rapport de maintenance préventive (planning, pièces de rechange)
    - Rapport d'incidents (historique anomalies par capteur)
    - Export PDF via WeasyPrint (optionnel) ou imprimable HTML
    - Rapport périodique automatique (quotidien / hebdomadaire / mensuel)
    - Lecture depuis realtime_results.json et anomaly_history_persist.json

Usage :
    python reporting_module.py --type daily
    python reporting_module.py --type weekly --sensor 8f7f2f7e
    python reporting_module.py --type full --output rapport_maintenance.html
    python reporting_module.py --serve   (serveur HTTP local pour visualiser)
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REPORT] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("reporting")

PROJECT_DIR  = Path(__file__).parent
RESULTS_FILE = PROJECT_DIR / "realtime_results.json"
HISTORY_FILE = PROJECT_DIR / "anomaly_history_persist.json"
ALERTS_FILE  = PROJECT_DIR / "alert_history.json"
REPORTS_DIR  = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  COLLECTE DE DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_entry(entry: dict) -> dict:
    """
    Normalise une entrée realtime_results.json vers un format plat uniforme.
    Supporte le format : {iteration, sensor_id, predict:{...}, rul:{...}, measurement:{...}}
    Et le format direct : {sensor_id, is_anomaly, risk_level, health_score, rul_hours, ...}
    """
    if "predict" not in entry and "rul" not in entry:
        return entry  # Déjà au format plat

    flat = {
        "sensor_id":   entry.get("sensor_id", "unknown"),
        "motor_id":    entry.get("motor_id"),
        "timestamp":   entry.get("timestamp", ""),
        "source":      entry.get("source", ""),
        "iteration":   entry.get("iteration", 0),
    }

    # Extraire depuis sous-objet predict
    predict = entry.get("predict", {}) or {}
    flat["prediction"]    = predict.get("prediction", "NORMAL")
    flat["is_anomaly"]    = predict.get("is_anomaly", False)
    flat["confidence"]    = predict.get("confidence", 0.0)
    flat["votes"]         = predict.get("votes", 0)
    flat["risk_level"]    = predict.get("risk_level", "OK")
    flat["anomaly_score"] = predict.get("anomaly_score", 0.0)

    # Health score : dans predict.features ou predict directement
    feat = predict.get("features", {}) or {}
    flat["health_score"]  = feat.get("health_score", predict.get("health_score", 0.0))

    # Extraire depuis sous-objet rul
    rul = entry.get("rul", {}) or {}
    flat["rul_hours"]      = rul.get("rul_hours", 0.0)
    flat["rul_days"]       = rul.get("rul_days", 0.0)
    flat["degradation_rate"] = rul.get("degradation_rate", 0.0)
    flat["alert_level"]    = rul.get("alert_level", "OK")

    # Mesures brutes
    meas = entry.get("measurement", {}) or {}
    flat["temperature"]   = meas.get("temperature")
    flat["vibration_z"]   = meas.get("vibration_z")
    flat["vibration_total"] = meas.get("vibration_total")
    flat["current"]       = meas.get("current")

    return flat


def load_results(max_entries: int = 500) -> list:
    """Charge et normalise les résultats temps réel depuis realtime_results.json."""
    if not RESULTS_FILE.exists():
        log.warning(f"Fichier résultats non trouvé : {RESULTS_FILE}")
        return []
    try:
        raw = RESULTS_FILE.read_text(encoding="utf-8").strip()
        all_entries = []

        # JSONDecoder stream : gère plusieurs JSON collés dans le même fichier
        import json as _json
        decoder = _json.JSONDecoder()
        pos = 0
        while pos < len(raw):
            try:
                obj, pos = decoder.raw_decode(raw, pos)
                if isinstance(obj, list):
                    all_entries.extend(obj)
                elif isinstance(obj, dict):
                    all_entries.append(obj)
                while pos < len(raw) and raw[pos] in ' \n\r\t,':
                    pos += 1
            except Exception:
                break

        if not all_entries:
            return []

        normalized = [_normalize_entry(e) for e in all_entries[-max_entries:]]
        return normalized

    except Exception as e:
        log.error(f"Erreur lecture résultats : {e}")
        return []


def load_anomaly_history() -> dict:
    """Charge l'historique des anomalies par capteur."""
    if not HISTORY_FILE.exists():
        return {}
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Erreur lecture historique : {e}")
        return {}


def load_alerts() -> list:
    """Charge l'historique des alertes envoyées."""
    if not ALERTS_FILE.exists():
        return []
    try:
        data = json.loads(ALERTS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return data.get("alerts", []) if isinstance(data, dict) else []
    except Exception as e:
        log.error(f"Erreur lecture alertes : {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  CALCUL DES KPIs
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(results: list, period_hours: float = 24.0) -> dict:
    """
    Calcule les KPIs de maintenance à partir des résultats de prédiction.

    Returns dict avec tous les indicateurs clés.
    """
    if not results:
        return {
            "total_predictions": 0, "anomaly_count": 0, "anomaly_rate_pct": 0,
            "avg_health_score": 0, "min_health_score": 0,
            "critical_alerts": 0, "warning_alerts": 0,
            "sensors_active": 0, "sensors_critical": [],
            "avg_rul_hours": 0, "min_rul_hours": 0,
            "period_hours": period_hours,
        }

    now = datetime.now()
    cutoff = now - timedelta(hours=period_hours)

    # Filtrer sur la période
    period_results = []
    for r in results:
        try:
            ts = datetime.fromisoformat(r.get("timestamp", ""))
            if ts >= cutoff:
                period_results.append(r)
        except Exception:
            period_results.append(r)  # Inclure si timestamp invalide

    if not period_results:
        period_results = results

    total = len(period_results)
    anomalies = [r for r in period_results if r.get("is_anomaly") or r.get("prediction") == "ANOMALY"]

    # Par capteur
    by_sensor = defaultdict(list)
    for r in period_results:
        sid = r.get("sensor_id", "unknown")
        by_sensor[sid].append(r)

    # Health scores
    health_scores = [r.get("health_score", r.get("features", {}).get("health_score", 0))
                     for r in period_results if r.get("health_score") or r.get("features", {}).get("health_score")]
    health_scores = [h for h in health_scores if h and h > 0]

    # RUL
    rul_values = [r.get("rul_hours", 0) for r in period_results if r.get("rul_hours", 0) > 0]

    # Capteurs critiques
    critical_sensors = []
    for sid, sensor_results in by_sensor.items():
        risk_levels = [r.get("risk_level", "OK") for r in sensor_results]
        if "CRITIQUE" in risk_levels or "ÉLEVÉ" in risk_levels:
            last = sensor_results[-1]
            critical_sensors.append({
                "sensor_id":    sid,
                "risk_level":   last.get("risk_level", "INCONNU"),
                "health_score": last.get("health_score", last.get("features", {}).get("health_score", 0)),
                "rul_hours":    last.get("rul_hours", 0),
                "anomaly_count": sum(1 for r in sensor_results if r.get("is_anomaly")),
            })

    anomaly_rate = (len(anomalies) / total * 100) if total > 0 else 0

    return {
        "total_predictions":  total,
        "anomaly_count":      len(anomalies),
        "anomaly_rate_pct":   round(anomaly_rate, 2),
        "avg_health_score":   round(np.mean(health_scores), 1) if health_scores else 0,
        "min_health_score":   round(np.min(health_scores), 1) if health_scores else 0,
        "critical_alerts":    sum(1 for r in period_results if r.get("risk_level") in ("CRITIQUE", "ÉLEVÉ")),
        "warning_alerts":     sum(1 for r in period_results if r.get("risk_level") == "MODÉRÉ"),
        "sensors_active":     len(by_sensor),
        "sensors_critical":   critical_sensors,
        "avg_rul_hours":      round(np.mean(rul_values), 1) if rul_values else 0,
        "min_rul_hours":      round(np.min(rul_values), 1) if rul_values else 0,
        "period_hours":       period_hours,
        "period_label":       f"Dernières {int(period_hours)}h" if period_hours < 48 else f"Derniers {int(period_hours/24)}j",
    }


def compute_sensor_summary(results: list) -> list:
    """Calcule un résumé par capteur."""
    by_sensor = defaultdict(list)
    for r in results:
        sid = r.get("sensor_id", "unknown")
        by_sensor[sid].append(r)

    summaries = []
    for sid, sensor_results in sorted(by_sensor.items()):
        last = sensor_results[-1]
        health_vals = [r.get("health_score", r.get("features", {}).get("health_score", 0))
                       for r in sensor_results]
        health_vals = [h for h in health_vals if h and h > 0]

        anomaly_count = sum(1 for r in sensor_results if r.get("is_anomaly"))
        anomaly_rate  = (anomaly_count / len(sensor_results) * 100) if sensor_results else 0

        rul_vals = [r.get("rul_hours", 0) for r in sensor_results if r.get("rul_hours", 0) > 0]
        last_rul = rul_vals[-1] if rul_vals else 0

        risk_level = last.get("risk_level", "OK")
        risk_class = {
            "CRITIQUE": "danger", "ÉLEVÉ": "warning",
            "MODÉRÉ": "info", "FAIBLE": "success", "OK": "success"
        }.get(risk_level, "secondary")

        summaries.append({
            "sensor_id":      sid,
            "motor_id":       last.get("motor_id", "—"),
            "risk_level":     risk_level,
            "risk_class":     risk_class,
            "health_score":   round(np.mean(health_vals), 1) if health_vals else 0,
            "anomaly_count":  anomaly_count,
            "anomaly_rate":   round(anomaly_rate, 1),
            "rul_hours":      round(last_rul, 0),
            "rul_days":       round(last_rul / 24, 1),
            "total_measures": len(sensor_results),
            "last_seen":      last.get("timestamp", "—"),
        })

    return sorted(summaries, key=lambda x: (
        {"CRITIQUE": 0, "ÉLEVÉ": 1, "MODÉRÉ": 2, "FAIBLE": 3, "OK": 4}.get(x["risk_level"], 5)
    ))


def compute_maintenance_schedule(sensor_summaries: list) -> list:
    """Génère le planning de maintenance préventive."""
    schedule = []
    now = datetime.now()

    for s in sensor_summaries:
        rul_h = s["rul_hours"]
        risk  = s["risk_level"]

        # Seuils CDC : CRITIQUE < 3j (72h), URGENT < 7j (168h), PLANIFIÉ < 14j (336h)
        if risk == "CRITIQUE" or rul_h < 72:
            due_date = now
            urgency  = "IMMÉDIAT"
            action   = "Arrêt moteur et remplacement roulement (RUL < 3 jours)"
            prio     = 1
        elif risk == "ÉLEVÉ" or rul_h < 168:
            due_date = now + timedelta(hours=max(1, rul_h * 0.5))
            urgency  = "URGENT"
            action   = "Planifier remplacement roulement sous 72h (RUL 3-7 jours)"
            prio     = 2
        elif risk == "MODÉRÉ" or rul_h < 336:
            due_date = now + timedelta(hours=max(24, rul_h * 0.7))
            urgency  = "PLANIFIÉ"
            action   = "Inspection et lubrification roulement sous 7 jours (RUL 7-14 jours)"
            prio     = 3
        else:
            due_date = now + timedelta(hours=rul_h * 0.8)
            urgency  = "ROUTINE"
            action   = "Vérification périodique standard (RUL > 14 jours)"
            prio     = 4

        schedule.append({
            "sensor_id":  s["sensor_id"],
            "motor_id":   s["motor_id"],
            "urgency":    urgency,
            "due_date":   due_date.strftime("%Y-%m-%d %H:%M"),
            "action":     action,
            "rul_hours":  s["rul_hours"],
            "health_score": s["health_score"],
            "priority":   prio,
        })

    return sorted(schedule, key=lambda x: x["priority"])


# ══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATION HTML
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --primary: #1a3a5c; --accent: #0088cc; --danger: #dc3545;
    --warning: #fd7e14; --success: #28a745; --info: #17a2b8;
    --bg: #f4f7fb; --card: #ffffff; --text: #2c3e50;
    --border: #dee2e6; --muted: #6c757d;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: var(--bg); color: var(--text); }}
  .header {{ background: linear-gradient(135deg, var(--primary), #2d6a9f); color: white;
             padding: 24px 32px; display: flex; justify-content: space-between; align-items: center; }}
  .header h1 {{ font-size: 24px; font-weight: 700; }}
  .header .meta {{ font-size: 13px; opacity: 0.85; text-align: right; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  .section {{ margin-bottom: 28px; }}
  .section-title {{ font-size: 16px; font-weight: 700; color: var(--primary);
                    border-left: 4px solid var(--accent); padding-left: 12px;
                    margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }}
  .kpi-card {{ background: var(--card); border-radius: 10px; padding: 20px; text-align: center;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 4px solid var(--accent); }}
  .kpi-card.danger  {{ border-top-color: var(--danger); }}
  .kpi-card.warning {{ border-top-color: var(--warning); }}
  .kpi-card.success {{ border-top-color: var(--success); }}
  .kpi-value {{ font-size: 36px; font-weight: 800; color: var(--primary); line-height: 1.1; }}
  .kpi-value.danger  {{ color: var(--danger); }}
  .kpi-value.warning {{ color: var(--warning); }}
  .kpi-value.success {{ color: var(--success); }}
  .kpi-label {{ font-size: 12px; color: var(--muted); margin-top: 6px; font-weight: 600; text-transform: uppercase; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card);
           border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  thead {{ background: var(--primary); color: white; }}
  th {{ padding: 12px 14px; font-size: 12px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.5px; text-align: left; }}
  td {{ padding: 11px 14px; font-size: 13px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f0f4f8; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px;
            font-weight: 700; text-transform: uppercase; }}
  .badge-danger  {{ background: #fde8ea; color: var(--danger); }}
  .badge-warning {{ background: #fff3e0; color: #e65100; }}
  .badge-info    {{ background: #e0f4ff; color: #0277bd; }}
  .badge-success {{ background: #e8f5e9; color: #1b5e20; }}
  .badge-secondary {{ background: #f0f0f0; color: #555; }}
  .alert-box {{ padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;
                border-left: 5px solid; display: flex; align-items: flex-start; gap: 12px; }}
  .alert-box.critical {{ background: #fde8ea; border-color: var(--danger); }}
  .alert-box.warning  {{ background: #fff8e1; border-color: var(--warning); }}
  .alert-box.info     {{ background: #e3f2fd; border-color: var(--info); }}
  .alert-box .alert-title {{ font-weight: 700; font-size: 14px; }}
  .alert-box .alert-body  {{ font-size: 13px; color: var(--muted); margin-top: 3px; }}
  .health-bar {{ height: 8px; border-radius: 4px; background: #e9ecef; overflow: hidden; }}
  .health-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .timeline {{ list-style: none; position: relative; padding-left: 24px; }}
  .timeline::before {{ content: ''; position: absolute; left: 8px; top: 0; bottom: 0;
                       width: 2px; background: var(--border); }}
  .timeline li {{ position: relative; padding: 0 0 18px 20px; }}
  .timeline li::before {{ content: ''; position: absolute; left: -8px; top: 4px;
                          width: 12px; height: 12px; border-radius: 50%;
                          background: var(--accent); border: 2px solid white;
                          box-shadow: 0 0 0 2px var(--accent); }}
  .timeline li.critical::before {{ background: var(--danger); box-shadow: 0 0 0 2px var(--danger); }}
  .timeline li.urgent::before   {{ background: var(--warning); box-shadow: 0 0 0 2px var(--warning); }}
  .footer {{ text-align: center; padding: 24px; color: var(--muted); font-size: 12px;
             border-top: 1px solid var(--border); margin-top: 32px; }}
  @media print {{
    .no-print {{ display: none !important; }}
    body {{ background: white; }}
    .kpi-card, table {{ box-shadow: none; border: 1px solid var(--border); }}
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Rapport de Maintenance Prédictive</h1>
    <div style="font-size:14px; opacity:0.9; margin-top:4px;">{subtitle}</div>
  </div>
  <div class="meta">
    Généré le {generated_at}<br>
    Système : PFE Maintenance Prédictive v5.1<br>
    Période : {period_label}
  </div>
</div>

<div class="container">

  <!-- RÉSUMÉ EXÉCUTIF -->
  <div class="section">
    <div class="section-title">Résumé Exécutif</div>
    <div class="kpi-grid">
      <div class="kpi-card {anomaly_card_class}">
        <div class="kpi-value {anomaly_val_class}">{anomaly_count}</div>
        <div class="kpi-label">Anomalies Détectées</div>
      </div>
      <div class="kpi-card {rate_card_class}">
        <div class="kpi-value {rate_val_class}">{anomaly_rate_pct}%</div>
        <div class="kpi-label">Taux d'Anomalie</div>
      </div>
      <div class="kpi-card {health_card_class}">
        <div class="kpi-value {health_val_class}">{avg_health_score}</div>
        <div class="kpi-label">Score Santé Moyen</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{sensors_active}</div>
        <div class="kpi-label">Capteurs Actifs</div>
      </div>
      <div class="kpi-card {critical_card_class}">
        <div class="kpi-value {critical_val_class}">{critical_alerts}</div>
        <div class="kpi-label">Alertes Critiques</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{total_predictions}</div>
        <div class="kpi-label">Prédictions Totales</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{avg_rul_hours}h</div>
        <div class="kpi-label">RUL Moyen</div>
      </div>
      <div class="kpi-card {min_rul_card_class}">
        <div class="kpi-value {min_rul_val_class}">{min_rul_hours}h</div>
        <div class="kpi-label">RUL Minimum</div>
      </div>
    </div>
  </div>

  <!-- ALERTES ACTIVES -->
  {active_alerts_section}

  <!-- PLANNING DE MAINTENANCE -->
  <div class="section">
    <div class="section-title">Planning de Maintenance Préventive</div>
    {schedule_html}
  </div>

  <!-- ÉTAT DES CAPTEURS -->
  <div class="section">
    <div class="section-title">État des Capteurs / Moteurs</div>
    {sensor_table_html}
  </div>

  <!-- HISTORIQUE DES ANOMALIES -->
  {anomaly_history_section}

</div>

<div class="footer">
  Rapport généré automatiquement par le système de maintenance prédictive PFE &mdash;
  {generated_at} &mdash; Toute décision de maintenance doit être validée par un technicien qualifié.
</div>

</body>
</html>"""


def _badge(text: str, style: str = "secondary") -> str:
    return f'<span class="badge badge-{style}">{text}</span>'


def _risk_badge(risk: str) -> str:
    styles = {"CRITIQUE": "danger", "ÉLEVÉ": "warning", "MODÉRÉ": "info",
              "FAIBLE": "success", "OK": "success"}
    return _badge(risk, styles.get(risk, "secondary"))


def _health_bar(score: float) -> str:
    color = "#dc3545" if score < 40 else "#fd7e14" if score < 70 else "#28a745"
    return (f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div class="health-bar" style="width:80px;">'
            f'<div class="health-fill" style="width:{score}%;background:{color};"></div></div>'
            f'<span style="font-size:12px;font-weight:600;">{score:.0f}</span></div>')


def _build_sensor_table(summaries: list) -> str:
    if not summaries:
        return '<p style="color:#6c757d;padding:16px;">Aucune donnée capteur disponible.</p>'

    rows = ""
    for s in summaries:
        rows += f"""<tr>
      <td><code style="font-size:12px;">{s['sensor_id']}</code></td>
      <td>{s.get('motor_id', '—')}</td>
      <td>{_risk_badge(s['risk_level'])}</td>
      <td>{_health_bar(s['health_score'])}</td>
      <td>{s['anomaly_count']} <span style="color:#6c757d;font-size:11px;">({s['anomaly_rate']}%)</span></td>
      <td style="font-weight:600;">{s['rul_hours']:.0f}h
        <span style="color:#6c757d;font-size:11px;">({s['rul_days']}j)</span></td>
      <td>{s['total_measures']}</td>
      <td style="font-size:11px;color:#6c757d;">{s['last_seen'][:19] if len(str(s['last_seen'])) > 10 else s['last_seen']}</td>
    </tr>"""

    return f"""<table>
  <thead><tr>
    <th>Capteur ID</th><th>Moteur</th><th>Risque</th><th>Santé</th>
    <th>Anomalies</th><th>RUL Estimé</th><th>Mesures</th><th>Dernière Vue</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _build_schedule_html(schedule: list) -> str:
    if not schedule:
        return '<p style="color:#6c757d;padding:16px;">Aucune maintenance planifiée.</p>'

    urgency_styles = {
        "IMMÉDIAT": ("critical", "danger"),
        "URGENT":   ("urgent",   "warning"),
        "PLANIFIÉ": ("",         "info"),
        "ROUTINE":  ("",         "success"),
    }

    rows = ""
    for item in schedule[:20]:  # Max 20 items
        li_class, badge_style = urgency_styles.get(item["urgency"], ("", "secondary"))
        rows += f"""<li class="{li_class}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <strong>{item['sensor_id']}</strong>
          {f"&nbsp;({item['motor_id']})" if item['motor_id'] != '—' else ''}
          &nbsp;{_badge(item['urgency'], badge_style)}
        </div>
        <div style="font-size:12px;color:#6c757d;">{item['due_date']}</div>
      </div>
      <div style="font-size:13px;margin-top:4px;color:#2c3e50;">{item['action']}</div>
      <div style="font-size:11px;color:#6c757d;margin-top:2px;">
        RUL : {item['rul_hours']:.0f}h — Santé : {item['health_score']:.0f}/100
      </div>
    </li>"""

    return f'<ul class="timeline">{rows}</ul>'


def _build_active_alerts_section(kpis: dict) -> str:
    critical = kpis.get("sensors_critical", [])
    if not critical:
        return """<div class="section">
      <div class="section-title">Alertes Actives</div>
      <div class="alert-box info">
        <div>
          <div class="alert-title">Aucune alerte critique</div>
          <div class="alert-body">Tous les capteurs fonctionnent dans les limites normales.</div>
        </div>
      </div>
    </div>"""

    alerts_html = ""
    for s in critical:
        style = "critical" if s["risk_level"] == "CRITIQUE" else "warning"
        alerts_html += f"""<div class="alert-box {style}">
      <div>
        <div class="alert-title">{s['risk_level']} — Capteur {s['sensor_id']}</div>
        <div class="alert-body">
          Santé : {s.get('health_score', 0):.0f}/100 &bull;
          RUL estimé : {s.get('rul_hours', 0):.0f}h &bull;
          Anomalies récentes : {s.get('anomaly_count', 0)}
        </div>
      </div>
    </div>"""

    return f"""<div class="section">
    <div class="section-title">Alertes Actives ({len(critical)})</div>
    {alerts_html}
  </div>"""


def _build_anomaly_history_section(history: dict) -> str:
    if not history:
        return ""

    rows = ""
    for sid, entries in sorted(history.items()):
        if not entries:
            continue
        recent = entries[-5:]
        avg_score = np.mean([e.get("score", 0) for e in entries])
        anomaly_count = sum(1 for e in entries if e.get("score", 0) >= 0.5)
        last_entry = entries[-1]
        last_ts = last_entry.get("timestamp", "—")[:19]
        rows += f"""<tr>
      <td><code style="font-size:12px;">{sid}</code></td>
      <td>{len(entries)}</td>
      <td>{anomaly_count}</td>
      <td>{avg_score:.3f}</td>
      <td>{last_ts}</td>
    </tr>"""

    if not rows:
        return ""

    return f"""<div class="section">
    <div class="section-title">Historique Anomalies (Mémoire Persistante)</div>
    <table>
      <thead><tr>
        <th>Capteur</th><th>Total Mesures</th><th>Anomalies</th>
        <th>Score Moyen</th><th>Dernière Mesure</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>"""


def generate_html_report(
    report_type: str = "daily",
    sensor_filter: Optional[str] = None,
    period_hours: float = None
) -> str:
    """
    Génère le rapport HTML complet.

    Args:
        report_type   : 'daily' (24h) | 'weekly' (168h) | 'monthly' (720h) | 'full'
        sensor_filter : Filtrer sur un capteur spécifique (None = tous)
        period_hours  : Période personnalisée en heures

    Returns : Contenu HTML du rapport.
    """
    period_map = {"daily": 24, "weekly": 168, "monthly": 720, "full": 1e9}
    if period_hours is None:
        period_hours = period_map.get(report_type, 24)

    type_labels = {
        "daily":   "Rapport Quotidien",
        "weekly":  "Rapport Hebdomadaire",
        "monthly": "Rapport Mensuel",
        "full":    "Rapport Complet",
    }

    log.info(f"Génération rapport {report_type} (période : {period_hours}h)")

    # Chargement des données
    results = load_results(max_entries=10000)
    history = load_anomaly_history()

    # Filtrage capteur
    if sensor_filter:
        results = [r for r in results if r.get("sensor_id") == sensor_filter]
        history = {k: v for k, v in history.items() if k == sensor_filter}
        subtitle = f"Capteur : {sensor_filter}"
    else:
        subtitle = f"Tous capteurs — {len(set(r.get('sensor_id','?') for r in results))} capteurs"

    # KPIs
    kpis = compute_kpis(results, period_hours=period_hours)
    sensor_summaries = compute_sensor_summary(results)
    schedule = compute_maintenance_schedule(sensor_summaries)

    # Classes CSS pour les KPIs
    def kpi_class(val, warn, crit, invert=False):
        if invert:
            return ("danger", "danger") if val <= crit else ("warning", "warning") if val <= warn else ("success", "success")
        return ("danger", "danger") if val >= crit else ("warning", "warning") if val >= warn else ("", "")

    ano_rate = kpis["anomaly_rate_pct"]
    anomaly_card, anomaly_val = kpi_class(kpis["anomaly_count"], 5, 20)
    rate_card, rate_val       = kpi_class(ano_rate, 10, 25)
    health_card, health_val   = kpi_class(kpis["avg_health_score"], 70, 50, invert=True)
    critical_card, critical_val = kpi_class(kpis["critical_alerts"], 1, 5)
    min_rul_h = kpis["min_rul_hours"]
    min_rul_card, min_rul_val = kpi_class(min_rul_h, 500, 100, invert=True)

    html = HTML_TEMPLATE.format(
        title=f"{type_labels.get(report_type, 'Rapport')} — Maintenance Prédictive",
        subtitle=subtitle,
        generated_at=datetime.now().strftime("%d/%m/%Y à %H:%M:%S"),
        period_label=kpis["period_label"],
        # KPIs
        anomaly_count=kpis["anomaly_count"],
        anomaly_rate_pct=kpis["anomaly_rate_pct"],
        avg_health_score=kpis["avg_health_score"],
        sensors_active=kpis["sensors_active"],
        critical_alerts=kpis["critical_alerts"],
        total_predictions=kpis["total_predictions"],
        avg_rul_hours=kpis["avg_rul_hours"],
        min_rul_hours=kpis["min_rul_hours"],
        # Classes CSS
        anomaly_card_class=anomaly_card, anomaly_val_class=anomaly_val,
        rate_card_class=rate_card,       rate_val_class=rate_val,
        health_card_class=health_card,   health_val_class=health_val,
        critical_card_class=critical_card, critical_val_class=critical_val,
        min_rul_card_class=min_rul_card,   min_rul_val_class=min_rul_val,
        # Sections
        active_alerts_section=_build_active_alerts_section(kpis),
        schedule_html=_build_schedule_html(schedule),
        sensor_table_html=_build_sensor_table(sensor_summaries),
        anomaly_history_section=_build_anomaly_history_section(history),
    )

    return html


def generate_json_report(report_type: str = "daily") -> dict:
    """Génère un rapport JSON (pour intégration API / automatisation)."""
    period_map = {"daily": 24, "weekly": 168, "monthly": 720, "full": 1e9}
    period_hours = period_map.get(report_type, 24)

    results = load_results()
    history = load_anomaly_history()

    kpis = compute_kpis(results, period_hours=period_hours)
    summaries = compute_sensor_summary(results)
    schedule = compute_maintenance_schedule(summaries)

    return {
        "report_type":    report_type,
        "generated_at":   datetime.now().isoformat(),
        "period_hours":   period_hours,
        "kpis":           kpis,
        "sensor_summaries": summaries,
        "maintenance_schedule": schedule[:10],
    }


def save_report(html_content: str, filename: str = None, report_type: str = "daily") -> Path:
    """Sauvegarde le rapport HTML dans le dossier reports/."""
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rapport_{report_type}_{ts}.html"

    out_path = REPORTS_DIR / filename
    out_path.write_text(html_content, encoding="utf-8")
    log.info(f"Rapport sauvegardé → {out_path}")
    return out_path


def export_pdf(html_path: Path) -> Optional[Path]:
    """Export PDF via WeasyPrint (si installé)."""
    try:
        from weasyprint import HTML as WP_HTML
        pdf_path = html_path.with_suffix(".pdf")
        WP_HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        log.info(f"PDF exporté → {pdf_path}")
        return pdf_path
    except ImportError:
        log.warning("WeasyPrint non installé — pip install weasyprint pour export PDF")
        return None
    except Exception as e:
        log.error(f"Erreur export PDF : {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SERVEUR HTTP LOCAL POUR VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def serve_report(port: int = 8080, report_type: str = "daily"):
    """Lance un serveur HTTP local pour visualiser le rapport en temps réel."""
    import http.server
    import threading
    import webbrowser

    class ReportHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Silencer les logs HTTP

        def do_GET(self):
            if self.path == "/" or self.path == "/report":
                html = generate_html_report(report_type)
                content = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(content))
                self.end_headers()
                self.wfile.write(content)
            elif self.path == "/api/report":
                data = generate_json_report(report_type)
                content = json.dumps(data, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Content-Length", len(content))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()

    server = http.server.HTTPServer(("", port), ReportHandler)
    url = f"http://localhost:{port}"
    log.info(f"Serveur de rapports démarré sur {url}")
    log.info(f"  Rapport HTML : {url}/report")
    log.info(f"  Rapport JSON : {url}/api/report")
    log.info("  Ctrl+C pour arrêter")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Serveur arrêté.")


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Générateur de rapports de maintenance")
    parser.add_argument("--type",   choices=["daily", "weekly", "monthly", "full"],
                        default="daily",  help="Type de rapport (défaut : daily)")
    parser.add_argument("--sensor", type=str, default=None,
                        help="Filtrer sur un capteur spécifique")
    parser.add_argument("--output", type=str, default=None,
                        help="Nom du fichier de sortie (défaut : auto)")
    parser.add_argument("--pdf",    action="store_true",
                        help="Exporter en PDF (nécessite WeasyPrint)")
    parser.add_argument("--json",   action="store_true",
                        help="Sortir le rapport en JSON")
    parser.add_argument("--serve",  action="store_true",
                        help="Lancer serveur HTTP local (port 8080)")
    parser.add_argument("--port",   type=int, default=8080,
                        help="Port du serveur HTTP (défaut : 8080)")
    args = parser.parse_args()

    if args.serve:
        serve_report(port=args.port, report_type=args.type)
        return

    if args.json:
        report = generate_json_report(args.type)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    html = generate_html_report(args.type, sensor_filter=args.sensor)
    path = save_report(html, filename=args.output, report_type=args.type)

    print(f"\nRapport généré : {path}")
    print(f"Ouvrir dans le navigateur : file:///{path.as_posix()}")

    if args.pdf:
        pdf_path = export_pdf(path)
        if pdf_path:
            print(f"PDF exporté  : {pdf_path}")

    # Afficher un résumé console
    results = load_results()
    kpis = compute_kpis(results, period_hours={"daily": 24, "weekly": 168, "monthly": 720}.get(args.type, 24))
    print(f"\nRésumé ({kpis['period_label']}) :")
    print(f"  Prédictions : {kpis['total_predictions']}")
    print(f"  Anomalies   : {kpis['anomaly_count']} ({kpis['anomaly_rate_pct']}%)")
    print(f"  Santé moy.  : {kpis['avg_health_score']}/100")
    print(f"  Critiques   : {kpis['critical_alerts']}")
    print(f"  Capteurs    : {kpis['sensors_active']}")


if __name__ == "__main__":
    main()
