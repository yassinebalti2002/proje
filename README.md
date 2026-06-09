# 🎯 PROJET FINAL — Système Maintenance Prédictive Temps Réel

**Version** : v5.1  
**Date** : May 20, 2026  
**Statut** : ✅ Production Ready  

---

## 📋 Vue d'Ensemble

Système complet de **détection d'anomalies et prédiction de durée de vie (RUL)** pour roulements industriels.

### Chaîne de Données
```
Capteurs IFM (19×)
    ↓ IO-Link
Gateway IFM (HTTP)
    ↓ INSERT
MariaDB full_data (192.168.1.50)
    ↓ SELECT (polling 2s)
Moteur realtime_mariadb.py
    ↓ POST /v1/predict + /v1/predict-rul
API FastAPI v3.1 (localhost:8000)
    ↓ Ensemble 4 modèles ML
Prédictions + RUL
    ↓
realtime_results.json + Dashboard
```

---

## 🚀 Démarrage Rapide

### 1️⃣ Installation (5 min)
```bash
python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2️⃣ Test Diagnostic
```bash
python realtime_mariadb.py --diagnostic
```

### 3️⃣ Lancer le Système

**Option A : Automatisé**
```bash
# Windows
demarrer_systeme.bat

# Linux/macOS
bash demarrer_systeme.sh
```

**Option B : Manuel — Terminal 1 (API)**
```bash
python api_unified_pythagore.py
```

**Option B : Manuel — Terminal 2 (Moteur)**
```bash
python realtime_mariadb.py
```

**Option C : Test Replay (pas besoin de capteurs)**
```bash
python realtime_mariadb.py --replay 50
```

---

## 📂 Structure du Projet

```
PROJET_FINAL/
│
├── 🐍 FICHIERS PYTHON
│   ├── api_unified_pythagore.py           (API FastAPI + prédictions)
│   ├── realtime_mariadb.py                (Moteur temps réel MariaDB)
│   ├── train_ecod_only.py                 (Entraînement ECOD — info)
│   ├── api_client.py                      (Client HTTP — test)
│   └── realtime_simulator.py              (Simulateur données — test)
│
├── 📊 DATA
│   ├── data/
│   │   └── dataset_2026_with_acc.csv      (1.2M lignes — entraînement)
│   └── models/                            ⭐ CRUCIAL
│       ├── model_if_v3.pkl                (Isolation Forest)
│       ├── model_lof_v3.pkl               (Local Outlier Factor)
│       ├── model_ocsvm_v3.pkl             (One-Class SVM)
│       ├── model_ecod_v3.pkl              (ECOD)
│       ├── scaler_v3.pkl                  (Normalisation)
│       ├── pca_v3.pkl                     (PCA 5 composantes)
│       ├── features_v3.pkl                (Liste 25 features)
│       └── metrics_v3.csv                 (F1=0.73, AUC=0.92)
│
├── 🌐 INTERFACE WEB
│   └── dashboard_realtime.html            (Vue temps réel)
│
├── 📄 CONFIGURATION
│   └── requirements.txt                   (Dépendances Python)
│
├── 📚 DOCUMENTATION ⭐
│   ├── QUICK_START.md                     (START HERE — 3 commandes)
│   ├── GUIDE_INSTALLATION_COLLEGUE.md     (Complet — 15 pages)
│   ├── CHECKLIST_COLLEGUE.md              (À cocher — validation)
│   ├── DEMARRAGE_RAPIDE.txt               (Notes)
│   └── ai_cp.sql                          (Schéma DB — info)
│
├── 🚀 SCRIPTS DE DÉMARRAGE
│   ├── demarrer_systeme.bat               (Windows)
│   ├── demarrer_systeme.sh                (Linux/macOS)
│   └── README.md                          (Ce fichier)
│
└── 📊 RÉSULTATS (générés à l'exécution)
    ├── realtime_results.json              (500 dernières prédictions)
    ├── realtime_mariadb.log               (Logs détaillés)
    ├── anomaly_history_persist.json       (Historique par capteur)
    └── tests/
        └── test_api_final.py              (Tests unitaires)
```

---

## ⚙️ Composants Clés

### 1. **API FastAPI** (`api_unified_pythagore.py`)

```python
# Endpoints principaux
POST /v1/predict            → Prédiction anomalies (4 modèles)
POST /v1/predict-rul        → Estimation RUL (Remaining Useful Life)
GET  /v1/health-score/{id}  → Score santé 0–100
GET  /health                → Health check
GET  /docs                  → Documentation interactive Swagger
```

**Features d'entrée** : 25 (température, vibrations X/Y/Z, courant, accélération, ratios)

**Modèles ML** :
- **IF** (Isolation Forest) — détecte anomalies globales
- **LOF** (Local Outlier Factor) — détecte déviations locales
- **OCSVM** (One-Class SVM) — hyperplan frontière
- **ECOD** (Empirical Cumulative Distribution) — distribution anormale

**Vote** : 2/4 modèles → Anomalie détectée

---

### 2. **Moteur Temps Réel** (`realtime_mariadb.py`)

**Fonction** : Lire MariaDB → Consolider données → Appeler API → Afficher résultats

**Polling** :
- Intervalle : 2 secondes (configurable)
- Batch : 100 lignes max par poll
- Fenêtre glissante : 10 mesures par capteur

**Consolidation** :
- Les capteurs envoient 3 types de mesures : temperature, vibration_x, vibration_y
- Regroupées par `MeasDetails.Id` pour former une session complète

**Sortie** :
```json
{
  "iteration": 42,
  "sensor_id": "8f7f2f7e",
  "measurement": {"temperature": 52.3, "vibration_z": 245.6},
  "predict": {
    "prediction": "ANOMALY",
    "votes": 3,
    "risk_level": "MODÉRÉ",
    "anomaly_score": 0.62
  },
  "rul": {
    "rul_hours": 48.5,
    "rul_days": 2.02,
    "health_score": 62,
    "alert_level": "ATTENTION"
  }
}
```

---

## 📊 25 Features Utilisées

| Catégorie | Features | Nombre |
|-----------|----------|--------|
| **Température** | mean, std, trend, current | 4 |
| **Vibration Z** | mean, std, rms, kurtosis, crest, current | 6 |
| **Vibration X** | mean, std, rms, kurtosis | 4 |
| **Vibration Y** | mean, std, rms, kurtosis | 4 |
| **Ratios inter-axes** | xy_ratio, xz_ratio | 2 |
| **Vibration 3D** | vib_total (√X²+Y²+Z²) | 1 |
| **Courant Moteur** | mean, std | 2 |
| **Score Santé** | health_score composite | 1 |
| **Accélération IFM** | acc_p2p, acc_z2p, acc_crest, acc_rms | 4 |
| **TOTAL** | | **25** |

---

## 🔄 Flux d'Exécution

```
┌─────────────────────────────────────────┐
│  MariaDB (SELECT id > last_id)          │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Consolidation 3 gph → Sessions         │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Buffer fenêtre glissante (10 mesures)  │
│  Par capteur                            │
└──────────────┬──────────────────────────┘
               ↓
        [Fenêtre pleine ?]
         Oui ↓         ↑ Non
             ↓         (attendre)
┌─────────────────────────────────────────┐
│  Extraction 25 features                 │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Scaler + PCA (5 composantes)           │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Vote Ensemble (IF + LOF + OCSVM + ECOD)│
│  Résultat : 0–4 votes                   │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Prédiction + RUL                       │
│  ANOMALY (≥2 votes) ou NORMAL           │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Affichage + Sauvegarde JSON            │
│  realtime_results.json                  │
└─────────────────────────────────────────┘
```

---

## 🎓 Modes de Lancement

### Mode 1 : Temps Réel (Données Capteurs)
```bash
python realtime_mariadb.py
```
- Attend les capteurs IFM
- Lit depuis MariaDB
- Prédictions en continu

### Mode 2 : Replay (Test sans Capteurs)
```bash
python realtime_mariadb.py --replay 50
```
- Rejeu des 50 dernières mesures stockées
- Rapide — parfait pour test local

### Mode 3 : Configuration Personnalisée
```bash
python realtime_mariadb.py \
  --host 192.168.1.50 \
  --window 20 \
  --poll 1.0 \
  --batch 50
```

---

## 🔧 Configuration MariaDB

### Connexion (3 options)

**Option 1 : Variables d'environnement** (recommandé — sécurisé)
```bash
export MARIADB_HOST=192.168.1.50
export MARIADB_USER=root
export MARIADB_PASSWORD=***
```

**Option 2 : CLI arguments**
```bash
python realtime_mariadb.py --host 192.168.1.50 --user root --password ***
```

**Option 3 : Éditer le code**
```python
# realtime_mariadb.py ligne 54
DEFAULT_CONFIG = {
    "host": "192.168.1.50",
    "user": "root",
    "password": "***",
}
```

---

## 📈 Métriques de Performance

### Modèles ML

| Métrique | Valeur | Source |
|----------|--------|--------|
| F1-Score | 0.73 | models/metrics_v3.csv |
| AUC | 0.92 | models/metrics_v3.csv |
| Precision | 0.71 | Détection précise |
| Recall | 0.75 | Couverture anomalies |
| Contamination | 5% | Baseline anomalies |

### Système Temps Réel

| Métrique | Valeur |
|----------|--------|
| Latence API | < 100ms |
| Prédictions/min | 2–3 |
| Fiabilité API | ≥ 90% |
| Mémoire | ~200 MB |

---

## 🌐 Endpoints API Détaillés

### `/v1/predict` (POST)
```python
Request:
{
  "sensor_id": "8f7f2f7e",
  "history": [
    {
      "temperature": 52.3,
      "vibration_x": 120.5,
      "vibration_y": 135.2,
      "vibration_z": 245.6,
      "current": 18.5,
      "acc_p2p": 0.45
    },
    ... (10 mesures)
  ]
}

Response:
{
  "prediction": "ANOMALY",
  "votes": 3,
  "anomaly_score": 0.62,
  "risk_level": "MODÉRÉ",
  "individual_models": {
    "IF": "ANOMALY",
    "LOF": "ANOMALY",
    "OCSVM": "NORMAL",
    "ECOD": "ANOMALY"
  }
}
```

### `/v1/predict-rul` (POST)
```python
Response:
{
  "rul_hours": 48.5,
  "rul_days": 2.02,
  "health_score": 62,
  "alert_level": "ATTENTION",
  "recommendation": "Surveiller la tendance vibratoire"
}
```

### `/health` (GET)
```python
{
  "status": "ok",
  "version": "3.1.0",
  "models": ["if", "lof", "ocsvm", "ecod"],
  "features": 25,
  "pca_components": 5
}
```

---

## 🚨 Niveaux de Risque

| Niveau | Condition | Action |
|--------|-----------|--------|
| **OK** | 0/4 votes | Surveillance standard |
| **FAIBLE** | 1/4 votes | Suivi normal |
| **MODÉRÉ** | 2/4 votes | Alerter opérateur |
| **ÉLEVÉ** | 3/4 votes | **Maintenance prévue** |
| **CRITIQUE** | 4/4 votes | **Arrêt immédiat** |

---

## 📞 Support & Troubleshooting

### Erreurs Fréquentes

| Erreur | Cause | Solution |
|--------|-------|----------|
| `ModuleNotFoundError` | Dépendance manquante | `pip install -r requirements.txt` |
| `Connection refused` (MariaDB) | Serveur injoignable | `ping 192.168.1.50` |
| `Access denied` | Credentials incorrects | Demander au IT IoT |
| `API port 8000 in use` | Port occupé | `netstat -an \| grep 8000` |
| `Models not loaded` | Fichiers pkl manquants | Vérifier dossier `models/` |

---

## 📚 Documentation Complète

**Lire dans cet ordre** :

1. ⭐ **QUICK_START.md** — 3 commandes pour démarrer
2. 📖 **GUIDE_INSTALLATION_COLLEGUE.md** — Installation détaillée
3. ✅ **CHECKLIST_COLLEGUE.md** — Validation étape par étape
4. 🔍 **Code commenté** — `api_unified_pythagore.py` & `realtime_mariadb.py`

---

## 🎯 Prochaines Étapes

### Court terme (Semaine 1)
- ✅ Installation réussie sur machine collègue
- ✅ Diagnostic MariaDB OK
- ✅ 50+ prédictions générées (mode replay ou temps réel)

### Moyen terme (Semaine 2–3)
- Valider prédictions avec données réelles
- Ajuster seuils risk_level si nécessaire
- Entraîner nouveau modèle si données suffisantes

### Long terme (Semaine 4+)
- Monitoring continu en production
- Alerte anomalies vers Slack/Email
- Dashboard web persistent
- Optimisation RUL avec historique réel

---

## 📞 Contact & Support

Pour questions ou problèmes :

- **Technique Python/ML** : Consulter docstrings dans le code
- **Réseau/MariaDB** : Contacter IT serveur IoT (192.168.1.50)
- **Déploiement** : Suivre `GUIDE_INSTALLATION_COLLEGUE.md`

---

## 📝 Licence & Versioning

**Version** : v5.1  
**Dernière mise à jour** : May 20, 2026  
**Statut** : ✅ Production Ready  

---

**✨ Bon test en temps réel ! 🚀**
