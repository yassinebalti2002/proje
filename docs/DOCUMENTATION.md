# Documentation complète — Système de Maintenance Prédictive

PFE — Mohamed Yassine Balti — ISG Bizerte × Novation City — 2025/2026

Ce document complète le `README.md` avec le détail technique complet : architecture, chaque
endpoint, pipeline ML de bout en bout, limites connues et historique des corrections apportées
lors de l'audit de code de juillet 2026.

---

## Table des matières

1. [Architecture générale](#1-architecture-générale)
2. [Structure du projet](#2-structure-du-projet)
3. [Authentification & sécurité](#3-authentification--sécurité)
4. [Endpoints API — référence complète](#4-endpoints-api--référence-complète)
5. [Pipeline ML — features, entraînement, inférence](#5-pipeline-ml--features-entraînement-inférence)
6. [Moteur temps réel](#6-moteur-temps-réel)
7. [Modules annexes](#7-modules-annexes)
8. [Limites connues](#8-limites-connues)
9. [Historique des corrections (audit juillet 2026)](#9-historique-des-corrections-audit-juillet-2026)

---

## 1. Architecture générale

```
Capteurs IFM (×19-20, IO-Link)
    │
    ▼
Gateway IoT ──► MariaDB (ai_cp.full_data — 1,6M+ lignes)
    │              chaque mesure physique = plusieurs lignes SQL
    │              (une par grandeur : temperature, vibration_x/y/z, acceleration)
    ▼
realtime_mariadb.py                    realtime_ifm_direct.py (alternative)
    │  polling SELECT WHERE id > last_id (2s)   │  lecture directe gateway HTTP
    │  consolide les lignes en sessions          │
    │  POST /v1/predict + /v1/predict-rul         │
    ▼                                             ▼
                    API FastAPI (api_unified_pythagore.py, port 8000)
                    │
                    ├─ auth.py          → X-API-Key sur tous les /v1/*
                    ├─ rate_limiter.py  → 429 par IP après N requêtes/60s
                    ├─ 6 modèles ML     → SoftVote (IF+ECOD+HBOS+COPOD)
                    ├─ RUL heuristique + GradientBoostingRegressor
                    ├─ alert_manager.py → email/webhook/SMS si CRITIQUE
                    └─ reporting_module.py → rapports HTML/JSON
                    │
                    ▼
            realtime_results.json (cache disque)
                    │
                    ▼
        Dashboard HTML5 (dashboard_predictive.html / dashboard_realtime.html)
        rafraîchissement 3s, servi par nginx (Docker) ou http.server (dev)
```

### Deux modes d'ingestion

| Mode | Fichier | Cas d'usage |
|---|---|---|
| MariaDB polling | `realtime_mariadb.py` | Production Novation City — lit la base alimentée par la gateway IFM |
| IFM direct | `realtime_ifm_direct.py` + `gateway_ifm_simulator.py` | Contournement si MariaDB indisponible — lit directement le HTTP de la gateway |
| Simulateur | `realtime_simulator.py` | Démo/dashboard sans données réelles — **ne pas utiliser pour valider le système** |

---

## 2. Structure du projet

```
api_unified_pythagore.py     API FastAPI — point d'entrée principal (2400+ lignes)
auth.py                      Authentification X-API-Key
rate_limiter.py              Rate limiting in-memory par IP
config.py                    Config centrale (MariaDB, IFM) — jamais commité (.gitignore)
signal_processing.py         FFT, enveloppe Hilbert, BPFO/BPFI/BSF, ondelettes
alert_manager.py             Alertes email SMTP / webhook Slack / SMS Twilio
reporting_module.py          Génération de rapports HTML/JSON
realtime_mariadb.py          Moteur temps réel — polling MariaDB
realtime_ifm_direct.py       Moteur temps réel — lecture directe gateway IFM
realtime_simulator.py        Simulateur de données (démo uniquement)
gateway_ifm_simulator.py     Émule les endpoints HTTP d'une vraie gateway IFM
api_client.py                Client HTTP partagé (retry, X-API-Key)
train_model_v3_unsupervised.py  Entraînement des 6 modèles de détection
train_rul_model.py           Entraînement du modèle RUL (GradientBoosting)
train_ecod_only.py           Script auxiliaire — OBSOLÈTE (schéma 25 features, garde-fou bloquant)
retrain_from_real_data.py    Script auxiliaire — formule health_score divergente, à ne pas utiliser tel quel
edge_optimize.py             Export/optimisation modèles pour déploiement embarqué (Raspberry Pi)
generate_dataset_from_sql.py Parsing d'un dump SQL → CSV
generate_architecture_diagram.py  Génère le diagramme d'architecture (image statique)
fix_mariadb.py               Script de patch ponctuel, déjà appliqué — non idempotent, ne pas relancer
dashboard_predictive.html    Dashboard SCADA HTML5 (mode predictif)
dashboard_realtime.html      Dashboard SCADA HTML5 (mode temps réel brut)
pipeline_upload.html         Interface web : upload SQL → parse → entraîne → recharge
tests/                       36 tests unitaires ML + 18 sécurité + 8 intégration
models/                      Artefacts entraînés (.pkl, scaler, PCA, seuils, métriques)
models_backup_before_leakage_fix/  Sauvegarde des modèles d'avant la correction du data leakage
```

---

## 3. Authentification & sécurité

### Clé API

Tous les endpoints `/v1/*` exigent l'en-tête `X-API-Key`. Les clés sont chargées **une fois au
démarrage** depuis la variable d'environnement `API_KEYS` (`.env`), séparées par des virgules.

```bash
API_KEYS=cle1,cle2,cle3
```

- Comparaison en **temps constant** (`hmac.compare_digest`) — résiste aux timing attacks.
- Si `API_KEYS` est vide/absent : **toutes** les requêtes `/v1/*` sont rejetées en 503 (fail-closed,
  pas un service ouvert par erreur).
- Endpoints publics (pas de clé requise) : `/`, `/health`, `/docs`, `/redoc`, `/metrics`, `/sensors`,
  `/anomalies`, `/v1/health-score/{id}`, `/v1/history/{id}`, `/v1/results`, `/v1/alert-level/{id}`.

### Rate limiting

`rate_limiter.py` implémente une fenêtre glissante en mémoire, par `(IP, chemin)`. Limites par
endpoint (voir `Depends(make_rate_limiter(N))` dans chaque route) : entre 5 et 60 requêtes/minute
selon la sensibilité de l'endpoint. Un balayage périodique (toutes les 5 min) purge les entrées
inactives depuis plus d'1h pour éviter une croissance non bornée du dictionnaire en mémoire.

**Limite connue** : le rate limiter utilise `request.client.host`, qui reflète l'IP du proxy si
l'API est déployée derrière un reverse proxy (ex. Render) sans `proxy_headers=True` configuré côté
uvicorn — dans ce cas, tous les clients externes partagent le même quota. Non corrigé (nécessite
un choix de configuration de déploiement propre à l'environnement cible).

### CORS

Contrôlé par `CORS_ORIGINS` (`.env`) : vide = ouvert à toutes origines (dev), sinon liste
d'origines séparées par des virgules (prod).

---

## 4. Endpoints API — référence complète

### Système

| Endpoint | Auth | Description |
|---|---|---|
| `GET /` | Non | Liste des endpoints disponibles |
| `GET /health` | Non | Statut + modèles chargés + nb capteurs en mémoire |
| `GET /metrics` | Non | Métriques ML officielles (lit `models/metrics_v3.csv`) |
| `GET /sensors` | Non | Liste des capteurs actifs (source : historique mémoire ou CSV) |
| `GET /anomalies` | Non | Anomalies filtrées par score minimum |
| `GET /v1/system-limits` | Oui | Documentation honnête des limites techniques connues |

### IA / Prédiction

| Endpoint | Auth | Description |
|---|---|---|
| `POST /v1/predict` | Oui | Détection d'anomalie — reçoit un historique de mesures, retourne le vote des 6 modèles |
| `POST /v1/predict-rul` | Oui | Estimation RUL (heuristique CDC + modèle ML GradientBoosting) |
| `POST /v1/iot-predict` | Oui | Predict + RUL en un seul appel, fenêtre glissante gérée côté serveur (pas besoin de renvoyer tout l'historique à chaque appel) |
| `POST /v1/spectral-analysis` | Oui | FFT, enveloppe, fréquences caractéristiques roulements (BPFO/BPFI/BSF) |
| `GET /v1/health-score/{id}` | Non | Score de santé 0-100, normalisé par baseline propre au capteur |
| `GET /v1/history/{id}` | Non | Historique des N dernières prédictions (RAM, reset au redémarrage) |
| `GET /v1/alert-level/{id}` | Non | Niveau d'alerte consolidé (OK/ATTENTION/URGENT/CRITIQUE) pour dashboard |
| `GET /v1/results` | Non | Dernières prédictions de tous les capteurs actifs |

### Reporting

| Endpoint | Auth | Description |
|---|---|---|
| `GET /v1/report` | Oui | Génère un rapport HTML ou JSON (daily/weekly/monthly/full) |

### Alertes

| Endpoint | Auth | Description |
|---|---|---|
| `GET /v1/alerts` | Oui | Historique des alertes envoyées |
| `GET /v1/alerts/stats` | Oui | Statistiques globales (total, par niveau, cooldowns actifs) |

### Pipeline (ré-entraînement via upload SQL)

| Endpoint | Auth | Description |
|---|---|---|
| `GET /pipeline` | Non | Page web d'upload d'un dump SQL |
| `POST /v1/pipeline/upload` | Oui | Upload `.sql` → parse → entraîne → recharge les modèles (asynchrone, thread background) |
| `GET /v1/pipeline/status/{job_id}` | Oui | Polling du statut d'un job de pipeline |
| `GET /v1/pipeline/jobs` | Oui | Liste des 10 derniers jobs |

### Format `/v1/predict`

```json
POST /v1/predict
Headers: X-API-Key: <clé>
{
  "sensor_id": "8f7f2f7e",
  "history": [
    { "temperature": 32.5, "vibration_x": 266.0, "vibration_y": 273.0, "vibration_z": 280.0 }
  ]
}
```

Réponse : `prediction` (NORMAL/ANOMALY), `anomaly_score` (0-1), `risk_level`
(FAIBLE/MODÉRÉ/ÉLEVÉ/CRITIQUE), `votes` (0-6), `individual_models` (détail par modèle),
`features` (les 31 features calculées, utile pour debug).

---

## 5. Pipeline ML — features, entraînement, inférence

### Les 31 features (extraites par fenêtre glissante de 20 mesures)

| Groupe | Features |
|---|---|
| Thermique (4) | temp_mean, temp_std, temp_trend, temp_cur |
| Vibration Z (6) | vib_z_mean/std/rms/kurt/crest/cur |
| Vibration X (4) | vib_x_mean/std/rms/kurt |
| Vibration Y (4) | vib_y_mean/std/rms/kurt |
| Globales (2) | vib_total (√(X²+Y²+Z²)), health_score |
| Accélération IFM (4) | acc_p2p, acc_z2p, acc_crest, acc_rms — **toujours à 0** (limitation matérielle gateway, voir §8) |
| Courant (1) | current_mean — **toujours à 0** (aucun capteur de courant installé) |
| Spectrales V6 (6) | delta_vib, delta_temp, vib_entropy, fft_ratio, vib_asym_xy, vib_asym_xz |

Filtre médian anti-spike (k=3) appliqué avant extraction. Fallback vib_x/vib_y sur vib_z si les
axes X/Y sont absents.

### Entraînement (`train_model_v3_unsupervised.py`)

1. Lecture directe MySQL `ai_cp.full_data` (25 000 sessions), fallback SQL dump ou
   `realtime_results.json` si MySQL indisponible.
2. Construction des fenêtres glissantes (stride=1, fenêtre=20) par capteur, labels heuristiques
   (score composite normalisé, seuil P90 → ~10% d'anomalies).
3. **Validation croisée PAR CAPTEUR ENTIER** (`GroupKFold`, 3 folds) — critique : les fenêtres
   glissantes se chevauchent à 95%, un split par ligne mettrait quasi la même fenêtre en train et
   en test (fuite de données). `GroupKFold` partitionne les 19 capteurs en 3 groupes disjoints :
   chaque capteur sert exactement une fois de test (jamais vu pendant l'entraînement de ce fold).
   Un split unique 80/20 s'est avéré à trop haute variance sur seulement 19 capteurs (un tirage
   malchanceux peut isoler presque toutes les anomalies d'un seul côté — voir §9) ; la moyenne sur
   3 folds donne une estimation bien plus stable.
4. Pour chaque fold : data augmentation ×3 (bruit gaussien 2%) sur le train uniquement,
   `RobustScaler`+PCA(0.95) fit sur le train uniquement, entraînement des 6 modèles, sélection du
   seuil SoftVote sous **contrainte precision ≥ 0.70** (`best_threshold_precision_f1`, filtre de
   persistance k=3) sur le train, métriques calculées sur le test du fold (jamais vu).
5. **Métriques officielles rapportées = moyenne des 3 folds** (généralisation honnête). Sauvegardées
   dans `metrics_v3.csv` avec `evaluation=test_holdout_par_capteur`.
6. **Modèle final déployé** : ré-entraîné sur l'**intégralité** des 19 capteurs (la CV ne sert qu'à
   estimer la performance, pas à produire l'artefact — pratique standard : plus de données
   disponibles = meilleur modèle final une fois l'approche validée par CV).

Vote SoftVote officiel : moyenne normalisée (MinMaxScaler fit sur train, transform sur test) de
IF + ECOD + HBOS + COPOD (LOF et OCSVM exclus du vote final — AUC individuels trop faibles,
gardés uniquement pour comparaison dans le tableau de diagnostic).

### Entraînement RUL (`train_rul_model.py`)

- Dataset synthétique : 150 moteurs simulés, courbes de dégradation de Weibull + bruit gaussien
  2-3%, calibrées sur les distributions réelles observées dans `full_data`.
- **Split train/test PAR MOTEUR ENTIER** (`GroupShuffleSplit` sur `motor_id`) — même raison que
  pour l'anomalie : deux points temporellement proches du même moteur sont quasi identiques, un
  split par ligne aurait laissé fuir de l'information (interpolation triviale).
- `GradientBoostingRegressor` (300 arbres, profondeur 5) sur 46 features (31 de base +
  15 spectrales FFT/enveloppe/BPFO-BPFI-BSF).
- Résultat (30 moteurs tenus à l'écart) : **MAE=294h (33%), R²=0.61**.

### Inférence (`api_unified_pythagore.py`)

- `extract_features()` — DOIT rester rigoureusement alignée avec le code d'entraînement
  (mêmes formules, même convention). Un point sensible : `safe_kurtosis()` utilise
  `fisher=False` (kurtosis "régulière", baseline≈3) pour matcher exactement
  `train_model_v3_unsupervised.py`. Une divergence ici décale silencieusement 3 des 31
  features par rapport à ce que les modèles ont appris (bug corrigé en juillet 2026, voir §9).
- `run_ensemble()` — calcule le score continu par modèle (`score_samples`/`decision_function`),
  normalise via p1/p99 stockés dans `threshold_v3.pkl`, moyenne pondérée SoftVote, filtre de
  persistance temporelle par capteur (k=3 fenêtres consécutives requises pour confirmer).
- `compute_rul()` — heuristique basée sur seuils industriels (temp/vib_total/kurtosis) +
  tendance de régression linéaire + historique d'anomalies en mémoire. Le RUL ML
  (`RULPredictor`) est utilisé pour **détecter une dégradation plus précoce** (peut élever le
  niveau d'alerte) mais les heures RUL rapportées restent celles de l'heuristique (calibrées sur
  les plages du cahier des charges).

### `/v1/iot-predict` — fenêtre serveur

Contrairement à `/v1/predict` (le client envoie tout l'historique à chaque appel), cet endpoint
maintient une fenêtre glissante **côté serveur** par capteur (`IOT_WINDOW_SIZE=20` — alignée avec
la taille de fenêtre d'entraînement, corrigée en juillet 2026, voir §9).

---

## 6. Moteur temps réel

### `realtime_mariadb.py`

- Polling `SELECT ... WHERE id > last_id` toutes les 2s (configurable `--poll`).
- Consolide les lignes multi-grandeurs (temperature/vibration_x/y/z/acceleration) en sessions
  complètes via un buffer `_pending` avec TTL de 60s (purge des sessions incomplètes).
- Envoie `X-API-Key` (lu depuis `API_KEYS` en environnement) sur chaque requête.
- Mode `--replay N` : rejoue les N dernières lignes réelles de la base (utile pour tester sans
  attendre de nouvelles données), sinon polling continu temps réel.

### `realtime_ifm_direct.py`

Variante qui lit directement les endpoints HTTP de la gateway IFM (`/getdata`, `/pdin`,
`/dataStorage`) sans passer par MariaDB. Utile en secours si la base est indisponible.

### `realtime_simulator.py`

Génère des données synthétiques réalistes (scénarios normal/dégradation/critique) et les poste à
l'API. **Usage démo/dashboard uniquement** — ne remplace pas un test avec de vraies données.

---

## 7. Modules annexes

### `alert_manager.py`

3 canaux : email SMTP, webhook (Slack/Teams), SMS (Twilio). Configuration dans
`alert_config.json` (jamais commiter une fois rempli avec des vrais identifiants — voir
`.gitignore` et `alert_config.example.json`). Cooldown par capteur (défaut 300s) : ne redéclenche
une alerte qu'après un envoi **réussi** sur au moins un canal (si tous les canaux échouent, le
cooldown n'est pas posé, pour permettre une nouvelle tentative rapide).

### `reporting_module.py`

Génère des rapports HTML/JSON (daily/weekly/monthly/full) à partir de `realtime_results.json`.
KPIs : santé moyenne/minimale, RUL moyen/minimal, taux d'anomalie, capteurs critiques.

### `signal_processing.py`

FFT, analyse d'enveloppe (démodulation Hilbert), fréquences caractéristiques de défauts
roulements (BPFO/BPFI/BSF/FTF, référence SKF 6205-2RS), décomposition en ondelettes (CWT
Morlet). Utilisé par `/v1/spectral-analysis` et enrichit les features RUL.

### `edge_optimize.py`

Export/optimisation des modèles pour déploiement embarqué (Raspberry Pi 4) : export ONNX,
benchmark, génération d'un `docker-compose.edge.yml` allégé.

---

## 8. Limites connues

Ces limites sont documentées de façon transparente (aussi exposées via `GET /v1/system-limits`) :

| # | Limite | Statut |
|---|---|---|
| L1 | Alertes email/SMS/webhook nécessitent une config manuelle (`alert_config.json`) | Module disponible, non activé par défaut |
| L2 | 4 features d'accélération (acc_p2p/z2p/crest/rms) toujours à 0 | Limitation matérielle — la gateway IFM AL1352 ne transmet pas ces valeurs de façon exploitable |
| L3 | Courant électrique (`current_mean`) toujours à 0 | Aucun capteur de courant installé sur le banc d'essai |
| L4 | Pas de déploiement Edge en production | `edge_optimize.py` prépare l'export mais n'est pas déployé |
| L5 | RUL heuristique, pas de modèle entraîné sur défaillances réelles | Aucun des 20 capteurs n'a atteint la défaillance complète pendant la collecte (nov 2025 → mai 2026) |
| L6 | Métriques anomalie encore modestes (F1=0,298) malgré la correction | 19 capteurs seulement → même avec GroupKFold (3 folds), le recall reste limité par le seuil sous contrainte precision≥0,70 ; AUC=0,9475 reste le signal fiable de la capacité de discrimination réelle du modèle |
| L7 | Durée d'entraînement variable | Un premier run a pris ~11h (contention système ponctuelle, non reproduite), un second run identique a pris 333s — pas un problème structurel du code, mais à surveiller si le ré-entraînement devient un processus récurrent automatisé |
| L8 | Rate limiting par IP peu fiable derrière un reverse proxy sans `proxy_headers` | Impact uniquement en déploiement Render/proxy, pas en local |
| L9 | `train_ecod_only.py` et `retrain_from_real_data.py` obsolètes | Schéma à 25 features (vs 31 actuel), formule `health_score` divergente de l'API — ne pas les utiliser pour ré-entraîner en production sans les mettre à jour d'abord |

---

## 9. Historique des corrections (audit juillet 2026)

Un audit de code complet a été mené sur l'ensemble du projet (12 000+ lignes). Corrections
appliquées, classées par catégorie :

### Sécurité / robustesse serveur

- **`realtime_simulator.py`, `realtime_ifm_direct.py`, `api_client.py`** : absence du header
  `X-API-Key` → 401 systématique contre une API protégée. Corrigé (lecture `API_KEYS` env).
- **`rate_limiter.py`** : le nettoyage anti-fuite mémoire était un no-op (suppression immédiatement
  suivie d'une recréation) → `_store` grossissait indéfiniment via des requêtes publiques à
  chemin variable (ex. `/v1/history/{id}` avec un id différent à chaque appel). Corrigé par un
  balayage périodique global (toutes les 5 min, purge des entrées inactives depuis 1h).
- **`realtime_mariadb.py`** : le TTL de purge des sessions incomplètes (`_pending`) n'était
  initialisé que sur la ligne `gph="temperature"` — une session ne recevant jamais cette ligne
  (capteur partiellement défaillant) n'était jamais purgée. Corrigé (TTL fixé dès la première
  ligne reçue, quel que soit son type).
- **`config.py`** : mot de passe MariaDB en clair comme seule source de vérité. Corrigé
  (surchargeable via variables d'environnement / `.env`, `load_dotenv()` ajouté).
- **`alert_config.json`** : suivi par Git sans figurer dans `.gitignore` → risque de fuite de
  secrets dès qu'un vrai mot de passe SMTP/token Twilio y serait renseigné. Corrigé
  (`.gitignore` + `alert_config.example.json`).

### Logique métier

- **`alert_manager.py`** : le cooldown par capteur était posé **avant** de savoir si l'envoi
  réussissait — si tous les canaux échouaient, aucune alerte ne repartait avant 300s pour un
  incident non notifié. Corrigé (cooldown posé uniquement en cas de succès d'au moins un canal).
  Statut SMS trompeur (un échec sur un destinataire faisait perdre le succès des autres) — corrigé.
- **`reporting_module.py`** : `health_score`/`rul_hours` égal à 0 (les pires cas réels) étaient
  exclus des KPI par un filtre `if h and h > 0` (0 est "falsy" en Python). Corrigé (`is not None`).
  Crash possible si `last_seen` n'est pas une chaîne (`int[:19]` invalide). Corrigé.
- **`edge_optimize.py`** : `anomaly_score` pouvait être négatif (seule la borne haute était
  clampée). Corrigé (`max(0.0, min(1.0, score))`).
- **`generate_dataset_from_sql.py`** : la dernière portion d'un fichier SQL ne se terminant pas
  par un retour à la ligne était silencieusement perdue (buffer résiduel jamais traité en fin de
  lecture). Corrigé.
- **`train_model_v3_unsupervised.py`** : métrique `Vote2/4`/`Vote3/4` toujours à 0 — mauvaise clé
  de dictionnaire (les clés réelles sont `Vote2/6`/`Vote3/6`, 6 modèles). Corrigé.

### Cohérence entraînement / inférence (impact modèle)

- **Convention de kurtosis divergente** : `train_model_v3_unsupervised.py` calculait
  `vib_z_kurt`/`vib_x_kurt`/`vib_y_kurt` avec `fisher=False` (baseline≈3), tandis que l'API
  utilisait `fisher=True` (baseline≈0) — 3 des 31 features étaient donc systématiquement décalées
  de ~3 unités par rapport à ce que les modèles avaient appris. **Corrigé** en alignant l'API sur
  la convention du training (`fisher=False`), avec ajustement en conséquence des seuils RUL
  heuristiques (`THRESHOLDS["vib_z_kurt"]`, décalés de +3).
- **Taille de fenêtre divergente** : `IOT_WINDOW_SIZE=10` côté API (`/v1/iot-predict`) contre
  `WINDOW_SIZE=20` à l'entraînement — biaisait std/trend/kurtosis/entropie calculés sur un
  échantillon deux fois plus petit que ce qu'ont vu les modèles. **Corrigé** (`IOT_WINDOW_SIZE=20`).

### Méthodologie ML (impact sur les métriques rapportées)

- **Data leakage / évaluation en resubstitution** : le `RobustScaler`/PCA/modèles étaient fit sur
  l'intégralité des données (originales + augmentées), puis évalués sur le sous-ensemble original
  — c'est-à-dire exactement les données qui avaient servi au fit. Les métriques rapportées
  (AUC=0.842, Recall=0.755, F1=0.629) étaient donc optimistes, pas représentatives d'une vraie
  généralisation. **Corrigé** en deux temps :
  1. Premier essai : split unique train/test par capteur entier (`GroupShuffleSplit`, 80/20).
     Résultat : **AUC=0.957** (signal fiable) mais **F1/Precision/Recall=0** — les 4 capteurs tenus
     à l'écart par ce tirage précis se sont avérés être parmi les plus sains du parc (8 anomalies
     réelles sur 4891 sessions, 0.2%, contre 10% dans la population globale). Symptomatique d'un
     split unique à haute variance sur un petit nombre de groupes (19 capteurs).
  2. Correction finale : **validation croisée par groupe** (`GroupKFold`, 3 folds — chaque capteur
     sert exactement une fois de test). Résultat stable et représentatif :
     **AUC=0.9475, F1=0.298, Precision=0.373, Recall=0.282** (moyenne des 3 folds). Le modèle
     réellement déployé est ensuite ré-entraîné sur l'intégralité des 19 capteurs — la CV a servi
     uniquement à estimer honnêtement sa performance de généralisation.
- **Split RUL par ligne au lieu du moteur** : `train_rul_model.py` faisait un
  `train_test_split` aléatoire ligne par ligne sur des courbes de dégradation continues par
  moteur — deux points temporellement proches du même moteur pouvaient se retrouver l'un en
  train, l'autre en test (fuite de données, interpolation triviale). **Corrigé**
  (`GroupShuffleSplit` par `motor_id`). Résultat après correction : **MAE=294h, R²=0.61**
  (meilleur que les anciens chiffres resubstitution : MAE=317h, R²=0.56).
- **Contrainte precision ≥ 0.70 jamais appliquée** : le code contenait une fonction
  `best_threshold_precision_f1(min_prec=0.70)` définie mais jamais appelée — le seuil réel était
  choisi par `best_threshold_f1` (sans contrainte), donnant une precision réelle de 0.539 malgré
  les commentaires affirmant le contraire. **Corrigé** : la fonction est maintenant réellement
  utilisée pour sélectionner le seuil SoftVote.

### Non corrigé intentionnellement (documenté comme limite)

- `train_ecod_only.py` et `retrain_from_real_data.py` restent divergents (formule health_score,
  schéma à 25 features) — scripts auxiliaires obsolètes, non utilisés par le pipeline principal.
- Rate limiting par IP derrière un reverse proxy sans `proxy_headers` — dépend du choix de
  déploiement, pas corrigé par défaut (voir L8).

Tous les fichiers modifiés ont été revalidés par la suite de tests complète (62/62 tests
passent) après chaque étape de correction, y compris après le ré-entraînement des modèles.
