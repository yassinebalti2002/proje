# Système Maintenance Prédictive — v3.1.0

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange?logo=scikitlearn)

> Détection d'anomalies en temps réel et estimation RUL pour 20 roulements industriels IFM — PFE ISG BIZERTE / Novation City

---

## Architecture complète

```
Capteurs IFM (19x) — IO-Link
        |
        v
  Gateway IFM (HTTP)
        |
        v
+----------------------------------------------+
|         API FastAPI v3.1.0                   |
|                                              |
|  +------------------+  +------------------+ |
|  | Ensemble 4 ML    |  | Traitement       | |
|  |                  |  | Signal           | |
|  | IF · LOF         |  |                  | |
|  | OCSVM · ECOD     |  | FFT · Spectral   | |
|  |                  |  | Bearing Fault    | |
|  | Vote 2/4         |  | Detection        | |
|  +------------------+  +------------------+ |
|                                              |
|  +------------------+  +------------------+ |
|  | Estimation RUL   |  | Alert Manager    | |
|  | GradientBoost    |  | Email/Webhook    | |
|  | + Heuristique    |  | SMS (Twilio)     | |
|  +------------------+  +------------------+ |
+----------------------------------------------+
        |
        v
  Dashboard HTML · JSON · Reporting
```

---

## Endpoints

| Methode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Accueil API |
| `GET` | `/health` | Statut + modeles charges |
| `GET` | `/metrics` | Metriques ML (F1, AUC, Accuracy) |
| `GET` | `/sensors` | Liste des 20 capteurs IFM |
| `GET` | `/anomalies` | Anomalies filtrees |
| `POST` | `/v1/predict` | Detection anomalie (4 modeles) |
| `POST` | `/v1/predict-rul` | Estimation RUL en heures |
| `POST` | `/v1/iot-predict` | Predict + RUL direct IoT sans BDD |
| `GET` | `/v1/health-score/{id}` | Score sante 0-100 par capteur |
| `GET` | `/v1/history/{id}` | Historique predictions capteur |
| `GET` | `/v1/alert-level/{id}` | Niveau d'alerte capteur |
| `POST` | `/v1/spectral-analysis` | Analyse FFT + defauts roulements |
| `GET` | `/v1/report` | Rapport HTML/JSON |
| `GET` | `/docs` | Swagger UI interactif |

---

## Modeles ML

### Detection d'anomalies (ensemble 4 modeles)

| Modele | Fichier | Role |
|--------|---------|------|
| Isolation Forest | `model_if_v3.pkl` | Points isoles |
| Local Outlier Factor | `model_lof_v3.pkl` | Densite locale |
| One-Class SVM | `model_ocsvm_v3.pkl` | Frontiere decision |
| ECOD | `model_ecod_v3.pkl` | Distribution empirique |

**Vote : 2/4 modeles -> anomalie detectee**

### Metriques

| Metrique | Valeur |
|----------|--------|
| F1 Score | 0.40 |
| AUC-ROC | 0.68 |
| Accuracy | 0.92 |

### Estimation RUL

| Propriete | Valeur |
|-----------|--------|
| Algorithme | GradientBoostingRegressor |
| Fichier | `model_rul_v1.pkl` |
| Features | 46 (spectrales + temporelles) |

---

## Installation locale

```bash
git clone https://github.com/yassinebalti2002/proje.git
cd proje
pip install -r requirements.txt
python api_unified_pythagore.py
```

API disponible sur `http://localhost:8000`

### Avec Docker

```bash
docker build -t maintenance-predictive .
docker run -p 8000:8000 maintenance-predictive
```

---

## Exemple d'utilisation

### POST /v1/iot-predict

```bash
curl -X POST http://localhost:8000/v1/iot-predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "capteur_01",
    "rms_x": 74.0,
    "rms_y": 70.0,
    "rms_z": 98.0,
    "temperature": 90.0,
    "current": 8.5
  }'
```

### Reponse type

```json
{
  "sensor_id": "capteur_01",
  "anomaly": true,
  "health_score": 25.0,
  "alert_level": "CRITICAL",
  "rul": {
    "hours": 48.0,
    "days": 2.0,
    "status": "ATTENTION — maintenance dans la semaine"
  },
  "votes": {
    "isolation_forest": "ANOMALIE",
    "lof": "ANOMALIE",
    "ocsvm": "ANOMALIE",
    "ecod": "normal"
  }
}
```

---

## Structure du projet

```
proje/
├── api_unified_pythagore.py    # API FastAPI principale (91 KB)
├── signal_processing.py        # FFT, spectral, defauts roulements
├── alert_manager.py            # Alertes email/webhook/SMS
├── train_rul_model.py          # Modele RUL GradientBoosting
├── reporting_module.py         # Generation rapports HTML/JSON
├── gateway_ifm_simulator.py    # Simulateur gateway IFM
├── requirements.txt
├── Dockerfile
├── render.yaml
└── models/
    ├── model_if_v3.pkl
    ├── model_lof_v3.pkl
    ├── model_ocsvm_v3.pkl
    ├── model_ecod_v3.pkl
    ├── model_rul_v1.pkl
    ├── scaler_v3.pkl
    └── pca_v3.pkl
```

---

## Modules optionnels

| Module | Requis | Si absent |
|--------|--------|-----------|
| MariaDB | Non | Endpoints IoT fonctionnent sans BDD |
| AlertManager | Non | Alertes externes desactivees |
| SignalProcessing | Non | Analyse spectrale desactivee |
| ReportingModule | Non | Rapports desactives |

---

## Niveaux d'alerte

| Score | Niveau | Action |
|-------|--------|--------|
| < 0.5 (0-1/4) | `OK` | Surveillance normale |
| >= 0.5 (2/4) | `WARNING` | Inspection preventive |
| >= 0.75 (3-4/4) | `CRITICAL` | Intervention urgente |

---

## Technologies

- **API** : FastAPI 0.115 + Uvicorn
- **ML** : scikit-learn 1.5.0, PyOD 2.0, SciPy
- **Signal** : FFT, analyse spectrale, detection defauts roulements (BPFO, BPFI, BSF)
- **Alertes** : Email SMTP, Webhook HTTP, SMS Twilio
- **Base de donnees** : MariaDB (optionnel)
- **Conteneurisation** : Docker
- **Deploiement** : Render.com

---

## Auteur

**Yassine Balti** — ISG BIZERTE
PFE Maintenance Predictive IoT — Novation City — 2025/2026
