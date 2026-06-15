# Système de Maintenance Prédictive des Roulements Industriels

**API FastAPI v3.1.0 · 6 modèles ML · IoT IFM · RUL GradientBoosting**

Projet de Fin d'Études — Mohamed Yassine Balti  
ISG Bizerte × Novation City (Sousse, Tunisie) — 2025/2026

---

## Vue d'ensemble

Système complet de **détection d'anomalies et de prédiction de durée de vie résiduelle (RUL)** pour 20 capteurs IFM industriels déployés sur les moteurs asynchrones de Novation City.

### Architecture

```
Capteurs IFM (×20)
    │  IO-Link
    ▼
Gateway IoT → MariaDB (ai_cp.full_data)
    │  polling SELECT WHERE id > last_id  (toutes les 2 s)
    ▼
realtime_mariadb.py
    │  POST /v1/predict + /v1/predict-rul
    ▼
API FastAPI v3.1.0  (port 8000 / $PORT)
    │  Ensemble 6 modèles · SoftVote seuil dynamique
    │  GradientBoostingRegressor (RUL · 46 features)
    ▼
Dashboard HTML5  (rafraîchissement 3 s)
```

---

## Performances ML réelles

| Métrique | Valeur | Note |
|----------|--------|------|
| AUC-ROC | **0,836** | Ensemble SoftVote |
| Recall | **0,877** | Priorité industrielle |
| Accuracy | 0,800 | 73 917 sessions |
| F1-Score | 0,367 | Contamination 20 % |
| Precision | 0,232 | |
| MAE RUL | **317 h** | GradientBoostingRegressor |
| R² RUL | **0,56** | 46 features spectrales |

---

## Modèles de détection (6)

| Modèle | Type | Rôle |
|--------|------|------|
| Isolation Forest | sklearn | Anomalies globales (arbres) |
| LOF | sklearn | Déviations locales (densité) |
| One-Class SVM | sklearn | Frontière hyperplan RBF |
| ECOD | pyod | Distribution empirique |
| HBOS | pyod | Histogramme par feature |
| COPOD | pyod | Copule multivariée |

Vote : **SoftVote à seuil dynamique optimal** (stocké dans `models/threshold_v3.pkl`)

---

## Démarrage rapide

### Pré-requis

- Python 3.11+
- pip

### Installation locale

```bash
git clone https://github.com/yassinebalti2002/proje.git
cd proje

python -m venv venv
source venv/bin/activate        # Linux/macOS
.\venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
```

### Lancer l'API

```bash
python api_unified_pythagore.py
```

Swagger interactif : http://localhost:8000/docs

### Mode replay (sans capteurs IFM)

```bash
python realtime_mariadb.py --replay 50
```

---

## Docker

### Build & Run

```bash
docker build -t maintenance-predictive .

# Run local
docker run -p 8000:8000 maintenance-predictive

# Run avec MariaDB local
docker run -p 8000:8000 \
  -e MARIADB_HOST=192.168.120.58 \
  -e MARIADB_PASSWORD=xxx \
  maintenance-predictive
```

### Docker Compose (API + Dashboard Nginx)

```bash
docker compose up
```

| Service | URL |
|---------|-----|
| API FastAPI | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Dashboard | http://localhost:3000/dashboard_realtime.html |

### Build ARM64 (Raspberry Pi 4)

```bash
docker buildx build --platform linux/arm64 -t maintenance-predictive:arm64 .
docker save maintenance-predictive:arm64 | ssh pi@<IP_RPI> docker load
```

---

## Déploiement Render

Ce dépôt inclut un fichier `render.yaml` prêt à l'emploi.

### Étapes

1. Se connecter sur [render.com](https://render.com) avec GitHub
2. **New → Web Service → Connect repository** → sélectionner `proje`
3. Render détecte automatiquement `render.yaml` et `Dockerfile`
4. Cliquer **Apply** — le build démarre (~3 min)
5. URL publique disponible : `https://maintenance-predictive-api.onrender.com`

> **Note :** Sur Render (plan gratuit), le service fonctionne en mode démo (replay)
> car MariaDB tourne sur le réseau local de Novation City.
> Pour connecter MariaDB, ajouter `MARIADB_HOST` et `MARIADB_PASSWORD`
> dans les variables d'environnement du dashboard Render.

### Variables d'environnement Render

| Variable | Valeur | Description |
|----------|--------|-------------|
| `PORT` | auto | Injecté automatiquement par Render |
| `DEMO_MODE` | `true` | Mode replay sans MariaDB |
| `MARIADB_HOST` | optionnel | IP/hostname MariaDB public |
| `MARIADB_PASSWORD` | optionnel | Mot de passe MariaDB |

---

## API — Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Statut API + modèles chargés |
| GET | `/metrics` | Métriques ML officielles |
| GET | `/sensors` | Liste des 20 capteurs IFM |
| GET | `/v1/dashboard/overview` | Résumé global tous capteurs |
| GET | `/v1/sensors/{id}/summary` | Résumé complet 1 capteur |
| GET | `/v1/alert-level/{id}` | Niveau d'alerte (FAIBLE/MODÉRÉ/ÉLEVÉ/CRITIQUE) |
| GET | `/v1/health-score/{id}` | Health Score 0–100 |
| GET | `/v1/history/{id}` | Historique 200 prédictions |
| GET | `/v1/live/{id}` | Fenêtre brute 20 mesures |
| POST | `/v1/predict` | Inférence 6 modèles SoftVote |
| POST | `/v1/predict-rul` | Estimation RUL (GBR) |

Documentation interactive : `GET /docs`

### Exemple `/v1/predict`

```json
POST /v1/predict
{
  "sensor_id": "68c11f06",
  "history": [
    { "temperature": 22.6, "vibration_x": 4.1, "vibration_y": 3.9,
      "vibration_z": 7.4, "current": 18.5, "acc_p2p": 0.12 }
  ]
}

// Réponse :
{
  "prediction": "NORMAL",
  "anomaly_score": 0.37,
  "risk_level": "MODÉRÉ",
  "votes": { "IF": 0, "LOF": 0, "OCSVM": 0, "ECOD": 1, "HBOS": 0, "COPOD": 0 },
  "health_score": 68.8
}
```

---

## Features ML (31)

| Groupe | Features | N |
|--------|----------|---|
| Température | mean, std, trend, cur | 4 |
| Vibration Z (axe critique) | mean, std, rms, kurtosis, crest, cur | 6 |
| Vibration X | mean, std, rms, kurtosis | 4 |
| Vibration Y | mean, std, rms, kurtosis | 4 |
| Globales | vib_total (Pythagore 3D), health_score | 2 |
| Accélération IFM | acc_p2p, acc_z2p, acc_crest, acc_rms | 4 |
| Courant | current_mean | 1 |
| **V6 spectrales** | delta_vib, delta_temp, vib_entropy, fft_ratio, vib_asym_xy, vib_asym_xz | **6** |
| **Total** | | **31** |

Fenêtre glissante : **w = 20**, pas **s = 3** — normalisées par `RobustScaler` + PCA 95 %

---

## Structure du projet

```
proje/
├── api_unified_pythagore.py        # API FastAPI v3.1.0 — point d'entrée principal
├── signal_processing.py            # FFT, enveloppe, BPFO/BPFI/BSF, wavelet
├── train_rul_model.py              # GradientBoostingRegressor RUL (46 features)
├── train_model_v3_unsupervised.py  # Entraînement 6 modèles (contamination 20 %)
├── realtime_mariadb.py             # Moteur temps réel — polling MariaDB 2 s
├── alert_manager.py                # Alertes email / webhook Slack
├── reporting_module.py             # Génération rapports HTML/JSON
├── dashboard_realtime.html         # Dashboard SCADA HTML5
├── requirements.txt                # Dépendances Python
├── Dockerfile                      # Image Docker multi-arch
├── docker-compose.yml              # API + Dashboard Nginx
├── render.yaml                     # Déploiement Render.com
└── models/
    ├── model_if_v3.pkl             # Isolation Forest
    ├── model_lof_v3.pkl            # Local Outlier Factor
    ├── model_ocsvm_v3.pkl          # One-Class SVM
    ├── model_ecod_v3.pkl           # ECOD (pyod)
    ├── model_hbos_v3.pkl           # HBOS (pyod)
    ├── model_copod_v3.pkl          # COPOD (pyod)
    ├── model_rul_v1.pkl            # GradientBoostingRegressor RUL
    ├── threshold_v3.pkl            # Seuil SoftVote optimal
    ├── metrics_v3.csv              # Métriques officielles
    └── metrics_rul_v1.json         # Métriques RUL
```

---

## Niveaux de risque

| Score anomalie | Niveau | Couleur | Action |
|----------------|--------|---------|--------|
| ≥ 0,75 | CRITIQUE | Rouge | Arrêt immédiat |
| 0,50–0,75 | ÉLEVÉ | Orange | Maintenance urgente |
| 0,25–0,50 | MODÉRÉ | Jaune | Surveillance renforcée |
| < 0,25 | FAIBLE | Vert | Normal |

Health Score : `H = 100 × (1 − 0,35·Tn − 0,35·Vn − 0,30·κn)` — conforme ISO 10816-3

---

## Entraînement des modèles

```bash
# 6 modèles de détection (contamination 20 %, w=20)
python train_model_v3_unsupervised.py

# Modèle RUL (GradientBoostingRegressor, Weibull synthétique)
python train_rul_model.py
```

---

## Configuration MariaDB

```bash
export MARIADB_HOST=192.168.120.58
export MARIADB_USER=root
export MARIADB_PASSWORD=<mot_de_passe>
export MARIADB_DATABASE=ai_cp
```

---

## Contact

**Mohamed Yassine Balti**  
PFE Licence Informatique de Gestion — ISG Bizerte / Université de Carthage  
Encadrant entreprise : M. Mohamed Chrifa (Novation City, Sousse)  
Email : balti.medyassine@gmail.com
