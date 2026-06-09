"""
alert_manager.py
================
Module de notifications externes pour le système de maintenance prédictive.

Canaux supportés :
  1. Email SMTP (Gmail, Outlook, serveur interne)
  2. Webhook HTTP (Slack, Microsoft Teams, Discord, Node-RED)
  3. SMS via API Twilio (optionnel — nécessite un compte Twilio)

Intégration :
    from alert_manager import AlertManager
    am = AlertManager()
    am.send_alert(sensor_id="91d92804", risk_level="CRITIQUE",
                  health_score=61.7, rul_hours=195, vib_total=1459)

Configuration :
    Créer un fichier alert_config.json dans le même dossier que ce script.
    Exemple minimal (email seulement) :
    {
        "email": {
            "enabled": true,
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "sender": "votre_email@gmail.com",
            "password": "mot_de_passe_app_gmail",
            "recipients": ["responsable@usine.tn", "maintenance@usine.tn"]
        }
    }

Génération du mot de passe d'application Gmail :
    Compte Google → Sécurité → Validation en 2 étapes → Mots de passe d'application
"""

import json
import logging
import smtplib
import threading
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

log = logging.getLogger("alert_manager")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION PAR DÉFAUT
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(__file__).parent / "alert_config.json"

DEFAULT_CONFIG = {
    "email": {
        "enabled": False,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "sender": "",
        "password": "",
        "recipients": []
    },
    "webhook": {
        "enabled": False,
        "url": "",        # URL Slack/Teams/Discord/Node-RED
        "type": "slack"   # "slack" | "teams" | "discord" | "generic"
    },
    "sms": {
        "enabled": False,
        "provider": "twilio",
        "account_sid": "",
        "auth_token": "",
        "from_number": "",
        "to_numbers": []
    },
    "rules": {
        # Niveaux déclenchant une notification (CRITIQUE obligatoire)
        "alert_levels": ["CRITIQUE", "URGENT"],
        # Délai minimum entre deux alertes pour le même capteur (secondes)
        "cooldown_seconds": 300,
        # Ne pas envoyer si health_score > seuil (évite les faux positifs)
        "min_health_threshold": 85
    }
}


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class AlertManager:
    """
    Gestionnaire d'alertes multi-canal pour le système de maintenance prédictive.

    Fonctionnement :
      - Les alertes sont envoyées en thread séparé (non bloquant pour l'API).
      - Un cooldown par capteur évite le spam en cas d'anomalie persistante.
      - Chaque alerte est loguée dans alert_history.json pour traçabilité.

    Exemple d'utilisation dans api_unified_pythagore.py :
        from alert_manager import AlertManager
        alert_mgr = AlertManager()

        # Dans l'endpoint /v1/predict, après run_ensemble() :
        if result["is_anomaly"] and risk in ("CRITIQUE", "ÉLEVÉ"):
            alert_mgr.send_alert(
                sensor_id=req.sensor_id,
                risk_level=risk,
                health_score=feat.get("health_score", 0),
                rul_hours=None,
                vib_total=feat.get("vib_total", 0),
                temperature=feat.get("temp_cur", 0),
                votes=result["votes"]
            )
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config = self._load_config(config_path)
        # {sensor_id: datetime} — dernière alerte envoyée par capteur
        self._last_alert: dict = {}
        # Historique persisté des alertes envoyées
        self._history_path = Path(__file__).parent / "alert_history.json"
        log.info("AlertManager initialisé — canaux actifs : " + self._active_channels())

    # ──────────────────────────────────────────────────────────────────────────
    #  Configuration
    # ──────────────────────────────────────────────────────────────────────────

    def _load_config(self, path: Path) -> dict:
        """Charge la config depuis alert_config.json, crée le fichier exemple si absent."""
        if not path.exists():
            path.write_text(
                json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            log.warning(
                f"Fichier de config absent → créé : {path}\n"
                "Modifie alert_config.json pour activer les notifications."
            )
            return DEFAULT_CONFIG.copy()
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            log.info(f"Config alertes chargée depuis {path}")
            return cfg
        except Exception as e:
            log.error(f"Erreur lecture config alertes : {e} — utilisation config par défaut")
            return DEFAULT_CONFIG.copy()

    def _active_channels(self) -> str:
        channels = []
        if self.config.get("email", {}).get("enabled"):
            channels.append("email")
        if self.config.get("webhook", {}).get("enabled"):
            channels.append("webhook")
        if self.config.get("sms", {}).get("enabled"):
            channels.append("SMS")
        return ", ".join(channels) if channels else "aucun (tous désactivés)"

    # ──────────────────────────────────────────────────────────────────────────
    #  Point d'entrée principal
    # ──────────────────────────────────────────────────────────────────────────

    def send_alert(
        self,
        sensor_id: str,
        risk_level: str,
        health_score: float,
        rul_hours: Optional[float],
        vib_total: Optional[float] = None,
        temperature: Optional[float] = None,
        votes: int = 0,
        extra: dict = None
    ):
        """
        Déclenche une alerte si les règles de filtrage sont satisfaites.
        L'envoi est effectué en arrière-plan (thread non bloquant).

        Paramètres
        ----------
        sensor_id    : ID du capteur IFM (ex: "91d92804")
        risk_level   : "CRITIQUE" | "URGENT" | "ÉLEVÉ" | "FAIBLE"
        health_score : Score de santé 0-100
        rul_hours    : RUL estimé en heures (None si non calculé)
        vib_total    : Vibration totale 3D en mg
        temperature  : Température actuelle en °C
        votes        : Nombre de modèles ayant voté ANOMALY (0-4)
        extra        : Dict d'informations supplémentaires optionnelles
        """
        rules = self.config.get("rules", DEFAULT_CONFIG["rules"])

        # Règle 1 — Niveau d'alerte suffisant
        if risk_level not in rules.get("alert_levels", ["CRITIQUE"]):
            return

        # Règle 2 — Health score sous le seuil (ne pas alerter les capteurs sains)
        min_health = rules.get("min_health_threshold", 85)
        if health_score > min_health:
            log.debug(f"Alerte ignorée — health_score={health_score} > seuil={min_health}")
            return

        # Règle 3 — Cooldown par capteur
        cooldown = rules.get("cooldown_seconds", 300)
        last = self._last_alert.get(sensor_id)
        if last and (datetime.now() - last).total_seconds() < cooldown:
            remaining = cooldown - (datetime.now() - last).total_seconds()
            log.debug(f"Alerte ignorée — cooldown {sensor_id} ({remaining:.0f}s restantes)")
            return

        # Préparer le contexte de l'alerte
        alert_ctx = {
            "sensor_id":   sensor_id,
            "risk_level":  risk_level,
            "health_score": health_score,
            "rul_hours":   rul_hours,
            "rul_days":    round(rul_hours / 24, 1) if rul_hours is not None else None,
            "vib_total":   vib_total,
            "temperature": temperature,
            "votes":       votes,
            "timestamp":   datetime.now().isoformat(),
            **(extra or {})
        }

        # Marquer le cooldown avant l'envoi (pour éviter les doublons rapides)
        self._last_alert[sensor_id] = datetime.now()

        # Lancer l'envoi en arrière-plan
        t = threading.Thread(
            target=self._dispatch_all,
            args=(alert_ctx,),
            daemon=True,
            name=f"alert-{sensor_id}"
        )
        t.start()
        log.info(f"🔔 Alerte déclenchée — capteur={sensor_id} niveau={risk_level}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _dispatch_all(self, ctx: dict):
        """Envoie sur tous les canaux activés et sauvegarde l'historique."""
        results = {}

        if self.config.get("email", {}).get("enabled"):
            results["email"] = self._send_email(ctx)

        if self.config.get("webhook", {}).get("enabled"):
            results["webhook"] = self._send_webhook(ctx)

        if self.config.get("sms", {}).get("enabled"):
            results["sms"] = self._send_sms(ctx)

        if not any(results.values()):
            log.warning(
                "Aucun canal d'alerte activé ou tous ont échoué. "
                "Vérifiez alert_config.json"
            )

        # Sauvegarder dans l'historique
        self._save_to_history(ctx, results)

    # ──────────────────────────────────────────────────────────────────────────
    #  Canal 1 — Email SMTP
    # ──────────────────────────────────────────────────────────────────────────

    def _send_email(self, ctx: dict) -> bool:
        """Envoie un email HTML via SMTP (Gmail, Outlook ou serveur interne)."""
        cfg = self.config.get("email", {})
        recipients = cfg.get("recipients", [])
        if not recipients:
            log.warning("Email activé mais aucun destinataire configuré")
            return False

        try:
            # Corps HTML
            subject = (
                f"[{ctx['risk_level']}] Capteur {ctx['sensor_id']} — "
                f"Maintenance prédictive"
            )
            body_html = self._build_email_html(ctx)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = cfg["sender"]
            msg["To"]      = ", ".join(recipients)
            msg.attach(MIMEText(self._build_email_text(ctx), "plain", "utf-8"))
            msg.attach(MIMEText(body_html, "html", "utf-8"))

            with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
                server.ehlo()
                server.starttls()
                server.login(cfg["sender"], cfg["password"])
                server.sendmail(cfg["sender"], recipients, msg.as_bytes())

            log.info(f"✅ Email envoyé → {recipients}")
            return True

        except smtplib.SMTPAuthenticationError:
            log.error(
                "Email : erreur d'authentification SMTP. "
                "Pour Gmail, utilisez un 'mot de passe d'application' "
                "(Compte Google → Sécurité → Mots de passe d'application)"
            )
        except smtplib.SMTPException as e:
            log.error(f"Email : erreur SMTP : {e}")
        except Exception as e:
            log.error(f"Email : erreur inattendue : {e}")
        return False

    def _build_email_text(self, ctx: dict) -> str:
        """Version texte brut de l'alerte (fallback email)."""
        rul_str = f"{ctx['rul_hours']:.0f}h ({ctx['rul_days']}j)" if ctx.get('rul_hours') else "N/A"
        return (
            f"ALERTE MAINTENANCE PRÉDICTIVE\n"
            f"{'='*40}\n"
            f"Capteur   : {ctx['sensor_id']}\n"
            f"Niveau    : {ctx['risk_level']}\n"
            f"Santé     : {ctx['health_score']}/100\n"
            f"RUL       : {rul_str}\n"
            f"Vibration : {ctx.get('vib_total', 'N/A')} mg\n"
            f"Temp.     : {ctx.get('temperature', 'N/A')}°C\n"
            f"Votes ML  : {ctx.get('votes', '?')}/4 modèles\n"
            f"Horodatage: {ctx['timestamp']}\n"
            f"{'='*40}\n"
            f"Dashboard : http://localhost:3000/dashboard_predictive.html\n"
            f"API       : http://localhost:8000/docs\n"
        )

    def _build_email_html(self, ctx: dict) -> str:
        """Version HTML stylisée de l'alerte email."""
        level_color = {
            "CRITIQUE": "#dc2626",
            "URGENT":   "#ea580c",
            "ATTENTION": "#ca8a04",
            "OK":       "#16a34a"
        }.get(ctx["risk_level"], "#6b7280")

        rul_str = (
            f"{ctx['rul_hours']:.0f}h ({ctx['rul_days']} jours)"
            if ctx.get("rul_hours") is not None else "Non disponible"
        )

        return f"""
<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width"></head>
<body style="font-family:Arial,sans-serif;background:#f3f4f6;padding:20px;margin:0;">
  <div style="max-width:600px;margin:0 auto;background:#fff;border-radius:12px;
              box-shadow:0 4px 12px rgba(0,0,0,0.1);overflow:hidden;">
    <!-- En-tête -->
    <div style="background:{level_color};color:#fff;padding:24px;text-align:center;">
      <h1 style="margin:0;font-size:24px;">⚠️ ALERTE MAINTENANCE</h1>
      <p style="margin:8px 0 0;font-size:18px;font-weight:bold;">{ctx['risk_level']}</p>
    </div>
    <!-- Corps -->
    <div style="padding:24px;">
      <h2 style="color:#1f2937;margin-top:0;">Capteur : <code>{ctx['sensor_id']}</code></h2>
      <table style="width:100%;border-collapse:collapse;">
        <tr style="background:#f9fafb;">
          <td style="padding:10px;border:1px solid #e5e7eb;font-weight:bold;">🏥 Score de santé</td>
          <td style="padding:10px;border:1px solid #e5e7eb;color:{level_color};font-weight:bold;">
            {ctx['health_score']}/100
          </td>
        </tr>
        <tr>
          <td style="padding:10px;border:1px solid #e5e7eb;font-weight:bold;">⏳ RUL estimé</td>
          <td style="padding:10px;border:1px solid #e5e7eb;">{rul_str}</td>
        </tr>
        <tr style="background:#f9fafb;">
          <td style="padding:10px;border:1px solid #e5e7eb;font-weight:bold;">📳 Vibration totale</td>
          <td style="padding:10px;border:1px solid #e5e7eb;">{ctx.get('vib_total', 'N/A')} mg</td>
        </tr>
        <tr>
          <td style="padding:10px;border:1px solid #e5e7eb;font-weight:bold;">🌡️ Température</td>
          <td style="padding:10px;border:1px solid #e5e7eb;">{ctx.get('temperature', 'N/A')} °C</td>
        </tr>
        <tr style="background:#f9fafb;">
          <td style="padding:10px;border:1px solid #e5e7eb;font-weight:bold;">🤖 Votes ML</td>
          <td style="padding:10px;border:1px solid #e5e7eb;">{ctx.get('votes', '?')}/4 modèles</td>
        </tr>
        <tr>
          <td style="padding:10px;border:1px solid #e5e7eb;font-weight:bold;">🕐 Horodatage</td>
          <td style="padding:10px;border:1px solid #e5e7eb;">{ctx['timestamp']}</td>
        </tr>
      </table>
      <!-- Recommandation -->
      <div style="background:#fef3c7;border-left:4px solid #f59e0b;
                  padding:16px;margin-top:20px;border-radius:4px;">
        <strong>Recommandation :</strong>
        {"Intervention immédiate requise. Vérifier le roulement et planifier l'arrêt machine."
         if ctx['risk_level'] == 'CRITIQUE'
         else "Surveillance renforcée. Planifier une inspection dans les 72h."}
      </div>
      <!-- Liens -->
      <div style="margin-top:20px;text-align:center;">
        <a href="http://localhost:3000/dashboard_predictive.html"
           style="background:{level_color};color:#fff;padding:12px 24px;
                  border-radius:8px;text-decoration:none;font-weight:bold;">
          Ouvrir le Dashboard
        </a>
      </div>
    </div>
    <!-- Pied de page -->
    <div style="background:#f9fafb;padding:16px;text-align:center;
                font-size:12px;color:#6b7280;">
      Système de Maintenance Prédictive — ISG Bizerte / Novation City<br>
      Ce message est généré automatiquement par l'API (port 8000).
    </div>
  </div>
</body>
</html>
"""

    # ──────────────────────────────────────────────────────────────────────────
    #  Canal 2 — Webhook HTTP (Slack / Teams / Discord / générique)
    # ──────────────────────────────────────────────────────────────────────────

    def _send_webhook(self, ctx: dict) -> bool:
        """
        Envoie un message vers un webhook HTTP.

        Formats supportés :
          - Slack  : https://hooks.slack.com/services/XXX
          - Teams  : https://xxx.webhook.office.com/webhookb2/...
          - Discord: https://discord.com/api/webhooks/...
          - Générique (Node-RED, tout serveur HTTP) : POST JSON brut
        """
        try:
            import requests
        except ImportError:
            log.error("requests non installé pour webhook")
            return False

        cfg = self.config.get("webhook", {})
        url = cfg.get("url", "")
        if not url:
            log.warning("Webhook activé mais URL vide dans la config")
            return False

        wtype = cfg.get("type", "generic").lower()
        level_emoji = {
            "CRITIQUE": "🔴", "URGENT": "🟠",
            "ATTENTION": "🟡", "OK": "🟢"
        }.get(ctx["risk_level"], "⚪")

        rul_str = (
            f"{ctx['rul_hours']:.0f}h ({ctx['rul_days']}j)"
            if ctx.get("rul_hours") is not None else "N/A"
        )

        try:
            if wtype == "slack":
                payload = {
                    "text": f"{level_emoji} *ALERTE {ctx['risk_level']}* — Capteur `{ctx['sensor_id']}`",
                    "attachments": [{
                        "color": {"CRITIQUE": "danger", "URGENT": "warning"}.get(ctx["risk_level"], "good"),
                        "fields": [
                            {"title": "Santé",      "value": f"{ctx['health_score']}/100", "short": True},
                            {"title": "RUL",        "value": rul_str,                      "short": True},
                            {"title": "Vibration",  "value": f"{ctx.get('vib_total','N/A')} mg", "short": True},
                            {"title": "Température","value": f"{ctx.get('temperature','N/A')}°C", "short": True},
                            {"title": "Votes ML",   "value": f"{ctx.get('votes','?')}/4",  "short": True},
                            {"title": "Horodatage", "value": ctx["timestamp"],              "short": False},
                        ],
                        "footer": "Maintenance Prédictive — ISG Bizerte"
                    }]
                }
            elif wtype == "teams":
                payload = {
                    "@type": "MessageCard",
                    "@context": "http://schema.org/extensions",
                    "themeColor": "dc2626" if ctx["risk_level"] == "CRITIQUE" else "ea580c",
                    "summary": f"Alerte {ctx['risk_level']} — Capteur {ctx['sensor_id']}",
                    "sections": [{
                        "activityTitle": f"{level_emoji} Alerte {ctx['risk_level']}",
                        "activitySubtitle": f"Capteur : {ctx['sensor_id']}",
                        "facts": [
                            {"name": "Santé",      "value": f"{ctx['health_score']}/100"},
                            {"name": "RUL",        "value": rul_str},
                            {"name": "Vibration",  "value": f"{ctx.get('vib_total','N/A')} mg"},
                            {"name": "Température","value": f"{ctx.get('temperature','N/A')}°C"},
                            {"name": "Votes ML",   "value": f"{ctx.get('votes','?')}/4"},
                            {"name": "Horodatage", "value": ctx["timestamp"]},
                        ]
                    }]
                }
            elif wtype == "discord":
                payload = {
                    "embeds": [{
                        "title": f"{level_emoji} Alerte {ctx['risk_level']} — Capteur {ctx['sensor_id']}",
                        "color": int("dc2626", 16) if ctx["risk_level"] == "CRITIQUE" else int("ea580c", 16),
                        "fields": [
                            {"name": "Santé",      "value": f"{ctx['health_score']}/100", "inline": True},
                            {"name": "RUL",        "value": rul_str,                      "inline": True},
                            {"name": "Vibration",  "value": f"{ctx.get('vib_total','N/A')} mg", "inline": True},
                            {"name": "Température","value": f"{ctx.get('temperature','N/A')}°C", "inline": True},
                        ],
                        "footer": {"text": f"ISG Bizerte | {ctx['timestamp']}"}
                    }]
                }
            else:
                # Payload générique — compatible Node-RED, serveur HTTP simple
                payload = {
                    "alert":       True,
                    "sensor_id":   ctx["sensor_id"],
                    "risk_level":  ctx["risk_level"],
                    "health_score": ctx["health_score"],
                    "rul_hours":   ctx.get("rul_hours"),
                    "vib_total":   ctx.get("vib_total"),
                    "temperature": ctx.get("temperature"),
                    "votes":       ctx.get("votes"),
                    "timestamp":   ctx["timestamp"],
                }

            r = requests.post(url, json=payload, timeout=10)
            if r.status_code in (200, 201, 204):
                log.info(f"✅ Webhook envoyé ({wtype}) → HTTP {r.status_code}")
                return True
            log.warning(f"Webhook : HTTP {r.status_code} — {r.text[:200]}")

        except Exception as e:
            log.error(f"Webhook : erreur envoi : {e}")
        return False

    # ──────────────────────────────────────────────────────────────────────────
    #  Canal 3 — SMS via Twilio
    # ──────────────────────────────────────────────────────────────────────────

    def _send_sms(self, ctx: dict) -> bool:
        """
        Envoie un SMS via l'API Twilio.
        Nécessite : pip install twilio
        Compte : https://www.twilio.com/try-twilio
        """
        cfg = self.config.get("sms", {})
        to_numbers = cfg.get("to_numbers", [])
        if not to_numbers:
            log.warning("SMS activé mais aucun numéro destinataire configuré")
            return False

        try:
            from twilio.rest import Client
        except ImportError:
            log.error("twilio non installé → pip install twilio")
            return False

        rul_str = f"{ctx['rul_hours']:.0f}h" if ctx.get("rul_hours") is not None else "N/A"
        body = (
            f"[ALERTE {ctx['risk_level']}] Capteur {ctx['sensor_id']} | "
            f"Santé: {ctx['health_score']}/100 | "
            f"RUL: {rul_str} | "
            f"Vib: {ctx.get('vib_total','N/A')}mg | "
            f"Temp: {ctx.get('temperature','N/A')}C"
        )

        try:
            client = Client(cfg["account_sid"], cfg["auth_token"])
            success = True
            for to in to_numbers:
                msg = client.messages.create(
                    body=body,
                    from_=cfg["from_number"],
                    to=to
                )
                log.info(f"✅ SMS envoyé → {to} | SID: {msg.sid}")
            return success
        except Exception as e:
            log.error(f"SMS Twilio : erreur : {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    #  Historique
    # ──────────────────────────────────────────────────────────────────────────

    def _save_to_history(self, ctx: dict, results: dict):
        """Sauvegarde l'alerte et son résultat dans alert_history.json."""
        record = {**ctx, "delivery": results}
        try:
            history = []
            if self._history_path.exists():
                history = json.loads(self._history_path.read_text(encoding="utf-8"))
            history.append(record)
            self._history_path.write_text(
                json.dumps(history[-200:], indent=2, ensure_ascii=False, default=str),
                encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"Historique alertes : échec sauvegarde : {e}")

    def get_history(self, limit: int = 50) -> list:
        """Retourne les dernières alertes envoyées (pour endpoint API)."""
        try:
            if self._history_path.exists():
                history = json.loads(self._history_path.read_text(encoding="utf-8"))
                return history[-limit:]
        except Exception:
            pass
        return []

    def get_stats(self) -> dict:
        """Statistiques sur les alertes envoyées."""
        history = self.get_history(limit=200)
        return {
            "total_alerts":    len(history),
            "by_level":        {
                lvl: sum(1 for h in history if h.get("risk_level") == lvl)
                for lvl in ["CRITIQUE", "URGENT", "ATTENTION"]
            },
            "active_cooldowns": {
                sid: (datetime.now() - last).total_seconds()
                for sid, last in self._last_alert.items()
                if (datetime.now() - last).total_seconds() < self.config.get("rules", {}).get("cooldown_seconds", 300)
            },
            "channels": self._active_channels(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  TEST EN LIGNE DE COMMANDE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Test rapide du gestionnaire d'alertes.
    Lance : python alert_manager.py
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [ALERT] %(message)s")

    print("Test AlertManager — maintenance prédictive")
    print("=" * 50)

    am = AlertManager()
    print(f"Canaux actifs : {am._active_channels()}")
    print(f"Config chargée depuis : {CONFIG_PATH}")
    print()

    # Simuler une alerte CRITIQUE
    print("Simulation alerte CRITIQUE pour capteur 91d92804...")
    am.send_alert(
        sensor_id   = "91d92804",
        risk_level  = "CRITIQUE",
        health_score= 61.7,
        rul_hours   = 195.1,
        vib_total   = 1459.0,
        temperature = 50.1,
        votes       = 4
    )

    # Attendre que le thread d'envoi termine
    time.sleep(3)

    print()
    print("Statistiques alertes :")
    print(json.dumps(am.get_stats(), indent=2, ensure_ascii=False, default=str))
    print()
    print("Pour activer les notifications, modifiez alert_config.json")
    print("Documentation : voir l'en-tête de ce fichier")
