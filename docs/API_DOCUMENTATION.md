# Documentation API — Maintenance Prédictive Roulements

**Fichier source** : `api_unified_pythagore.py`
**Version API** : `3.1.0`
**Base URL (local)** : `http://localhost:8000`
**Documentation interactive** : `/docs` (Swagger UI) · `/redoc` (ReDoc)

Cette documentation couvre **uniquement l'API HTTP** — pour l'architecture du système,
le pipeline ML ou les modules annexes, voir `DOCUMENTATION.md`.

---

## Sommaire

1. [Authentification](#1-authentification)
2. [Rate limiting](#2-rate-limiting)
3. [CORS](#3-cors)
4. [Modèles de données communs](#4-modèles-de-données-communs)
5. [Codes d'erreur](#5-codes-derreur)
6. [Référence des endpoints](#6-référence-des-endpoints)
   - [Système](#système)
   - [IA / Prédiction](#ia--prédiction)
   - [Reporting](#reporting)
   - [Alertes](#alertes)
   - [Pipeline (ré-entraînement)](#pipeline-ré-entraînement)
7. [Tableau récapitulatif](#7-tableau-récapitulatif)

---

## 1. Authentification

Implémentée dans `auth.py`. Système de clé API à 2 rôles (RBAC minimal).

### En-tête requis

```
X-API-Key: <votre_clé>
```

Le nom de l'en-tête est configurable via la variable d'environnement `API_KEY_HEADER`
(défaut : `X-API-Key`).

### Rôles

| Rôle | Variable d'environnement | Droits |
|---|---|---|
| **admin** | `API_KEYS=cle1,cle2` | Tous les endpoints protégés, y compris `/v1/pipeline/upload` |
| **operator** | `API_KEYS_OPERATOR=cle3,cle4` | Tous les endpoints protégés **sauf** `/v1/pipeline/upload` |

Les clés sont comparées en **temps constant** (`hmac.compare_digest`) pour résister aux
timing attacks. Génération d'une clé sûre :

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Comportement

- En-tête absent → **401** `En-tête X-API-Key manquant.`
- Clé invalide → **401** `Clé API invalide.`
- Clé opérateur sur un endpoint admin-only → **403** `Cette opération nécessite une clé API de niveau admin.`
- Aucune clé configurée côté serveur (`API_KEYS` et `API_KEYS_OPERATOR` vides) → **503** (fail-closed : l'API refuse tout plutôt que de s'ouvrir par erreur)

Chaque tentative (succès/échec) est journalisée dans `audit.log` (JSON Lines) avec un
timestamp, le résultat (`GRANTED`/`DENIED`/`FORBIDDEN`), le rôle, l'IP, la méthode, le
chemin, et une **empreinte SHA-256 tronquée** de la clé (jamais la clé en clair).

### Endpoints publics (sans clé)

`GET /`, `GET /health`, `/docs`, `/redoc`, `GET /metrics`, `GET /v1/model-card`,
`GET /sensors`, `GET /anomalies`, `GET /v1/health-score/{id}`, `GET /v1/history/{id}`,
`GET /v1/alert-level/{id}`, `GET /v1/results`, `GET /v1/spectral/{id}`, `GET /pipeline`,
`GET /v1/alerts`, `GET /v1/alerts/stats`.

> Le détail exact « auth requise ou non » par endpoint est indiqué dans le
> [tableau récapitulatif](#7-tableau-récapitulatif).

---

## 2. Rate limiting

Implémenté dans `rate_limiter.py` — fenêtre glissante en mémoire, thread-safe, sans
dépendance externe (pas de Redis).

- Clé de comptage : `(IP source, chemin)` — et pour les `POST`, le `sensor_id` extrait du
  corps JSON est ajouté à la clé (`ip:chemin:sensor_id`), pour que chaque capteur ait son
  propre quota plutôt qu'un quota partagé par IP sur `/v1/predict`.
- Limite dépassée → **429** avec en-tête `Retry-After: <secondes>` et message
  `Trop de requetes. Limite : N appels/60s par IP. Reessayez dans Xs.`
- Purge automatique des entrées inactives depuis plus d'1h (toutes les 5 min) pour éviter
  une fuite mémoire.
- La limite par minute varie selon l'endpoint (5 à 60 req/min) — voir le tableau
  récapitulatif.

**Limite connue** : basé sur `request.client.host`, qui reflète l'IP du reverse proxy si
l'API est déployée derrière un proxy sans `proxy_headers=True` configuré côté uvicorn
(tous les clients externes partageraient alors le même quota).

---

## 3. CORS

Contrôlé par la variable d'environnement `CORS_ORIGINS` :
- Non définie → `*` (toutes origines, adapté au dev)
- Définie → liste d'origines séparées par des virgules (à utiliser en production)

Méthodes autorisées : `GET`, `POST`. En-têtes : tous.

---

## 4. Modèles de données communs

### `MeasurePoint` — une mesure capteur

Tous les champs sont optionnels sauf mention contraire par endpoint.

| Champ | Type | Plage physique | Description |
|---|---|---|---|
| `timestamp` | string (ISO 8601) | — | Horodatage de la mesure |
| `temperature` | float | -20 à 150 °C | Température |
| `vibration_x` | float | 0 à 5000 mg | Vibration RMS axe X |
| `vibration_y` | float | 0 à 5000 mg | Vibration RMS axe Y |
| `vibration_z` | float | 0 à 5000 mg | Vibration RMS axe Z |
| `current` | float | 0 à 500 A | Courant moteur |
| `power` | float | 0 à 100 000 W | Puissance |
| `vitesse` | float | 0 à 10 000 RPM | Vitesse |
| `a_rms` | float | 0 à 5000 | — |
| `crest` | float | 0 à 50 | — |
| `acc_p2p` | float | 0 à 30 000 mg | Accélération peak-to-peak (axe Y) |
| `acc_z2p` | float | 0 à 15 000 mg | Accélération zero-to-peak (axe Y) |
| `acc_crest` | float | 0 à 5000 | Facteur de crête accélération |
| `acc_rms` | float | 0 à 5000 mg | Accélération RMS (axe Y) |

> Toute valeur hors plage déclenche une erreur de validation Pydantic (**422**).

### Dictionnaire `features` (retourné par `/v1/predict`, `/v1/iot-predict`)

Calculé par `extract_features()` à partir de l'historique fourni. Principales clés :

- **Thermique** : `temp_mean`, `temp_std`, `temp_trend`, `temp_cur`
- **Vibration Z** : `vib_z_mean`, `vib_z_std`, `vib_z_rms_w`, `vib_z_kurt`, `vib_z_crest`, `vib_z_cur`
- **Vibration X** : `vib_x_mean`, `vib_x_std`, `vib_x_rms_w`, `vib_x_kurt`
- **Vibration Y** : `vib_y_mean`, `vib_y_std`, `vib_y_rms_w`, `vib_y_kurt`
- **Ratios inter-axes** : `vib_xy_ratio`, `vib_xz_ratio`
- **Courant** : `current_mean`, `current_std` (toujours 0 — aucun capteur de courant installé, voir `/v1/system-limits`)
- **Vibration totale (Pythagore 3D)** : `vib_total = √(X² + Y² + Z²)` — base ISO 10816-3 / ISO 20816
- **Santé** : `health_score` (0–100)
- **Accélération IFM** : `acc_p2p`, `acc_z2p`, `acc_crest`, `acc_rms` (souvent 0 — limitation gateway)
- **V6 (avancées)** : `delta_vib`, `delta_temp`, `vib_entropy` (entropie de Shannon), `fft_ratio` (périodicité anormale), `vib_asym_xy`, `vib_asym_xz`

Un filtre médian anti-spike (k=3) est appliqué avant extraction sur température et vibration.

---

## 5. Codes d'erreur

| Code | Signification | Exemple de cause |
|---|---|---|
| `400` | Requête invalide | `history` vide, moins de mesures que le minimum requis |
| `401` | Non authentifié | En-tête `X-API-Key` absent ou clé invalide |
| `403` | Authentifié mais non autorisé | Clé opérateur sur un endpoint admin-only |
| `404` | Ressource introuvable | Capteur sans historique, `job_id` inconnu |
| `422` | Erreur de validation Pydantic | Champ hors plage physique, type incorrect |
| `429` | Trop de requêtes | Rate limit dépassé (voir en-tête `Retry-After`) |
| `500` | Erreur serveur | Exception non gérée pendant le traitement |
| `503` | Service indisponible | Aucune clé API configurée, module optionnel non chargé (`signal_processing`, `reporting_module`) |

Toutes les réponses JSON sont servies en `application/json; charset=utf-8` (forcé
explicitement pour éviter la corruption des accents sous PowerShell 5.1).

---

## 6. Référence des endpoints

### Système

#### `GET /`
Auth : non · Rate limit : aucun
Retourne l'identité de l'API et la liste résumée des endpoints principaux.

#### `GET /health`
Auth : non · Rate limit : aucun
Health check — état des modèles chargés.

```json
{
  "status": "ok",
  "models_loaded": true,
  "models": ["if", "lof", "ocsvm", "ecod", "hbos", "copod"],
  "features_count": 31,
  "version": "3.1.0",
  "n_sensors_in_memory": 19,
  "timestamp": "2026-08-07T10:00:00"
}
```

#### `GET /metrics`
Auth : non · Rate limit : 30/min
Métriques de performance du modèle (source : `models/metrics_v3.csv`) : `f1_score`,
`accuracy`, `precision`, `recall`, `auc_roc`, `n_anomalies`, `n_total`, `contamination`,
`weights` (poids appris du stacking par modèle), `window_size`, `n_features`.

#### `GET /v1/model-card`
Auth : non · Rate limit : 30/min
« Model card » — provenance des données d'entraînement, méthodologie, limites connues,
pour chaque composant (détection d'anomalie et RUL). Utile pour auditer la fiabilité
d'une prédiction avant de l'utiliser pour une décision critique.

#### `GET /v1/system-limits`
Auth : **oui** (opérateur ou admin) · Rate limit : 30/min
Liste structurée (L1 à L6) des limitations techniques connues : alertes non
persistantes par défaut, features d'accélération toujours nulles, absence de capteur de
courant, pas de déploiement edge, RUL heuristique non validé sur pannes réelles, etc.

---

### IA / Prédiction

#### `POST /v1/predict`
Auth : **oui** · Rate limit : 60/min (par IP + `sensor_id`)

Détection d'anomalie en temps réel via un ensemble non supervisé
**IF + LOF + OCSVM + ECOD + HBOS + COPOD**, fusionné par **stacking LogisticRegression**
(remplace l'ancienne moyenne fixe). Un filtre de **persistance temporelle** (k=3 fenêtres
consécutives par défaut) confirme l'anomalie avant de la remonter, pour réduire les faux
positifs isolés.

**Requête** (`PredictRequest`) :

```json
{
  "sensor_id": "8f7f2f7e",
  "motor_id": "Motor_1604",
  "history": [
    { "temperature": 32.5, "vibration_x": 266.0, "vibration_y": 273.0, "vibration_z": 280.0, "current": 0 }
  ]
}
```
`history` : minimum 1 mesure (`MeasurePoint`).

**Réponse** (`PredictResponse`, 200) :

```json
{
  "sensor_id": "8f7f2f7e",
  "motor_id": "Motor_1604",
  "timestamp": "2026-08-07T10:00:00",
  "prediction": "NORMAL",
  "is_anomaly": false,
  "confidence": 0.12,
  "votes": 1,
  "risk_level": "FAIBLE",
  "anomaly_score": 0.12,
  "individual_models": { "IF": "NORMAL", "LOF": "ANOMALY", "OCSVM": "NORMAL", "ECOD": "NORMAL", "HBOS": "NORMAL", "COPOD": "NORMAL" },
  "individual_scores": { "IF": 0.08, "LOF": 0.61, "OCSVM": 0.15, "ECOD": 0.10, "HBOS": 0.09, "COPOD": 0.11 },
  "features": { "temp_mean": 32.5, "vib_total": 391.2, "health_score": 94.3, "...": "..." }
}
```

Champs clés :
- `prediction` : `"NORMAL"` ou `"ANOMALY"` (après filtre de persistance)
- `risk_level` : `FAIBLE` (< 0.25) · `MODÉRÉ` (< 0.50) · `ÉLEVÉ` (< 0.75) · `CRITIQUE` (≥ 0.75) — recalculé en `FAIBLE` si `health_score ≥ 85` et pas d'anomalie confirmée
- `votes` : nombre de modèles (0–6) dont le score individuel dépasse 0.5
- `individual_models` / `individual_scores` : détail par modèle, y compris LOF/OCSVM (diagnostic uniquement — poids appris quasi nul dans le stacking si bruités)

Effets de bord : met à jour l'historique glissant du capteur (pour le RUL et
`/v1/history`), alimente le buffer brut vibration (pour `/v1/spectral/{id}`), et
déclenche une alerte externe (email/webhook/SMS) si `risk_level` est `CRITIQUE` ou
`ÉLEVÉ` et qu'`AlertManager` est configuré.

Erreurs : `400` si `history` vide.

---

#### `POST /v1/predict-rul`
Auth : **oui** · Rate limit : 60/min

Estime le temps restant avant défaillance (**RUL — Remaining Useful Life**), en heures et
en jours, en combinant :
1. Score de dégradation instantané (position vs seuils industriels : température,
   `vib_total`, kurtosis)
2. Tendance temporelle (régression linéaire sur vibration/température)
3. Historique des anomalies récentes du capteur (mémoire serveur)
4. Un modèle ML dédié (`GradientBoostingRegressor`, si chargé) qui peut **aggraver**
   (jamais atténuer) le niveau d'alerte déterminé par l'heuristique

> ⚠️ Le RUL est une **estimation empirique** — aucun modèle supervisé sur pannes réelles
> confirmées n'est disponible à ce jour (voir `/v1/model-card`). Ne pas utiliser comme
> seule base d'une décision d'arrêt machine.

**Requête** (`RULRequest`) :

```json
{
  "sensor_id": "8f7f2f7e",
  "motor_id": "Motor_1604",
  "prediction": "NORMAL",
  "votes": 1,
  "confidence": 0.12,
  "risk_level": "FAIBLE",
  "anomaly_score": 0.12,
  "history": [
    { "temperature": 32.5, "vibration_x": 266.0, "vibration_y": 273.0, "vibration_z": 280.0 },
    { "temperature": 33.1, "vibration_x": 270.0, "vibration_y": 275.0, "vibration_z": 285.0 },
    { "temperature": 33.8, "vibration_x": 272.0, "vibration_y": 278.0, "vibration_z": 290.0 }
  ]
}
```
`history` : **minimum 3 mesures** (nécessaire pour calculer une tendance). Les champs
`prediction`/`votes`/`confidence`/`risk_level`/`anomaly_score` sont typiquement ceux
retournés par un appel préalable à `/v1/predict` pour le même capteur.

**Réponse** (`RULResponse`, 200) :

```json
{
  "sensor_id": "8f7f2f7e",
  "motor_id": "Motor_1604",
  "timestamp": "2026-08-07T10:00:00",
  "rul_hours": 456.2,
  "rul_days": 19.01,
  "degradation_rate": 18.4,
  "health_score": 94.3,
  "confidence": "MOYENNE",
  "alert_level": "OK",
  "recommendation": "Fonctionnement normal. RUL > 14 jours. Prochaine inspection planifiée selon calendrier.",
  "trend": {
    "temp_trend": 0.42,
    "vib_total_trend": 3.1,
    "vib_formula": "sqrt(X² + Y² + Z²)",
    "deg_instant": 0.21,
    "deg_rate": 0.05,
    "hist_anomaly_factor": 0.0,
    "rul_model": "heuristic_CDC + ML_gradient_boosting"
  }
}
```

- `alert_level` : `OK` (>14j) · `ATTENTION` (7–14j) · `URGENT` (3–7j) · `CRITIQUE` (<3j)
- `confidence` : `HAUTE` (≥10 mesures) · `MOYENNE` (≥5) · `FAIBLE` (<5)
- Garde-fou capteur sain : `health_score ≥ 85` force `alert_level = OK` avec un RUL
  plancher de 336h (14 jours), pour éviter les faux `URGENT` observés en pratique.

Erreurs : `400` si moins de 3 mesures dans `history`.

---

#### `POST /v1/iot-predict`
Auth : **oui** · Rate limit : 60/min

Endpoint pensé pour la **production sans accès base de données**. Le client envoie
**une mesure à la fois** ; le serveur maintient une **fenêtre glissante de 20 mesures**
par `sensor_id` (`IOT_WINDOW_SIZE`) et retourne **prédiction d'anomalie + RUL en un seul
appel**. Le RUL n'est calculé qu'à partir de 3 mesures accumulées côté serveur (`null`
avant).

**Requête** (`IoTMeasurementRequest`) :

```json
{
  "sensor_id": "8f7f2f7e",
  "motor_id": "Motor_8f7f2f7e",
  "temperature": 32.5,
  "vibration_x": 266.0,
  "vibration_y": 273.0,
  "vibration_z": 280.0,
  "current": 0.0
}
```
`temperature`, `vibration_x`, `vibration_y`, `vibration_z` sont **requis** ;
`timestamp` (généré si absent), `current`, `acc_p2p`, `acc_rms`, `acc_crest`, `acc_z2p`
sont optionnels.

**Réponse** (`IoTPredictResponse`, 200) — union des champs de `/v1/predict` et
`/v1/predict-rul`, plus `window_size` (nombre de mesures actuellement en mémoire pour ce
capteur) :

```json
{
  "sensor_id": "8f7f2f7e",
  "motor_id": "Motor_8f7f2f7e",
  "timestamp": "2026-08-07T10:00:00",
  "window_size": 4,
  "prediction": "NORMAL",
  "is_anomaly": false,
  "confidence": 0.10,
  "votes": 0,
  "risk_level": "FAIBLE",
  "anomaly_score": 0.10,
  "individual_models": { "...": "..." },
  "individual_scores": { "...": "..." },
  "rul_hours": 500.0,
  "rul_days": 20.83,
  "health_score": 96.1,
  "alert_level": "OK",
  "recommendation": "Fonctionnement normal...",
  "features": { "...": "..." }
}
```

---

#### `POST /v1/spectral-analysis`
Auth : **oui** · Rate limit : 20/min

Analyse spectrale complète du signal de vibration : FFT (spectre de puissance,
fréquences dominantes), analyse d'enveloppe (démodulation Hilbert), fréquences
caractéristiques de défauts roulements (**BPFO, BPFI, BSF, FTF** — roulement SKF
6205-2RS), décomposition en ondelettes (CWT Morlet).

**Requête** : identique à `PredictRequest` (voir `/v1/predict`), plus deux paramètres
query optionnels : `rpm` (défaut 1450.0), `fs` (fréquence d'échantillonnage Hz, défaut
100.0). Minimum **8 mesures** `vibration_z` dans `history`.

```
POST /v1/spectral-analysis?rpm=1450&fs=100
```

**Réponse** : `spectral_features`, `bearing_analysis` (défauts détectés par fréquence
caractéristique), `wavelet`, `metadata`, `ml_feature_vector`.

Erreurs : `400` si moins de 8 mesures `vibration_z` ; `503` si `signal_processing.py`
non disponible (dépendance `scipy` manquante).

---

#### `GET /v1/spectral/{sensor_id}`
Auth : non · Rate limit : 30/min

Version **publique et en lecture seule** de l'analyse spectrale : utilise le buffer
serveur des dernières valeurs `vibration_z` déjà reçues via `/v1/predict` pour ce
capteur (pas de recalcul côté client, pensé pour le dashboard).

Paramètres query : `rpm` (1450.0), `fs` (100.0).

```json
{ "available": false, "reason": "Pas assez de mesures en buffer (3/8 min)." }
```
ou, si assez de données (≥8 mesures) :
```json
{
  "available": true,
  "sensor_id": "8f7f2f7e",
  "signal_length": 64,
  "analysis_params": { "fs_hz": 100.0, "rpm": 1450.0 },
  "spectral_features": { "...": "..." },
  "bearing_analysis": { "...": "..." },
  "raw_spectra": { "...": "..." },
  "metadata": { "...": "..." }
}
```

---

#### `GET /v1/health-score/{sensor_id}`
Auth : non · Rate limit : 60/min

Score de santé (0–100) **normalisé par baseline propre au capteur** (calculée sur ses 20
premières mesures) — évite le biais d'une comparaison à un seuil global identique pour
tous les moteurs.

```json
{
  "sensor_id": "8f7f2f7e",
  "health_score": 91.4,
  "health_raw": 88.0,
  "baseline": 0.12,
  "score_method": "relatif_baseline",
  "anomaly_rate": 0.1,
  "n_records": 42,
  "last_score": 0.09,
  "trend": "STABLE",
  "timestamp": "2026-08-07T10:00:00"
}
```
`score_method` vaut `relatif_baseline` une fois la baseline établie, sinon
`brut_en_attente_baseline`. `trend` : `DÉGRADATION` / `AMÉLIORATION` / `STABLE`.

Si le capteur n'a aucun historique, retourne `health_score: 100.0` avec `n_records: 0`
(pas d'erreur 404 sur cet endpoint).

---

#### `GET /v1/history/{sensor_id}`
Auth : non · Rate limit : 60/min

Historique glissant des prédictions d'un capteur (en mémoire — **réinitialisé au
redémarrage de l'API**, sauf persistance disque `anomaly_history_persist.json`
restaurée au démarrage pour les capteurs IFM connus).

Paramètre query : `limit` (défaut 20) — nombre d'entrées les plus récentes à retourner.
Maximum en mémoire : **50 entrées** par capteur (`HISTORY_WINDOW`).

```json
{
  "sensor_id": "8f7f2f7e",
  "n_total": 50,
  "n_returned": 20,
  "avg_score": 0.14,
  "max_score": 0.62,
  "anomaly_rate": 0.1,
  "trend": "STABLE",
  "trend_value": 0.0012,
  "history": [ { "timestamp": "2026-08-07T09:58:00", "score": 0.11, "confidence": 0.11 } ],
  "timestamp": "2026-08-07T10:00:00"
}
```

Erreurs : `404` si aucun historique pour ce capteur (invite à appeler `POST /v1/predict`
au préalable).

---

#### `GET /v1/alert-level/{sensor_id}`
Auth : non · Rate limit : 60/min

Niveau d'alerte **consolidé** sur les 10 dernières prédictions, pensé pour alimenter un
feu tricolore de dashboard.

```json
{
  "sensor_id": "8f7f2f7e",
  "alert_level": "OK",
  "color": "green",
  "icon": "🟢",
  "avg_score": 0.12,
  "anomaly_rate": 0.0,
  "trend": "→ STABLE",
  "n_measures": 10,
  "message": "Fonctionnement nominal.",
  "timestamp": "2026-08-07T10:00:00"
}
```
`alert_level` : `OK` (vert) · `ATTENTION` (jaune) · `URGENT` (orange) · `CRITIQUE`
(rouge) · `INCONNU` (gris, si aucune donnée).

---

#### `GET /v1/results`
Auth : non · Rate limit : 60/min

Dernières prédictions (anomalie **+** RUL fusionnées) de tous les capteurs actifs, ou
d'un capteur précis via `?sensor_id=8f7f2f7e`. Mis à jour à chaque appel `/v1/predict` ou
`/v1/predict-rul`. Endpoint `GET` simple, accessible directement depuis un navigateur.

Erreurs : `404` si `sensor_id` fourni mais introuvable dans le cache.

---

### Reporting

#### `GET /v1/report`
Auth : **oui** · Rate limit : 20/min

Génère un rapport de maintenance à partir des données temps réel.

Paramètres query :
- `type` : `daily` (24h, défaut) · `weekly` (7j) · `monthly` (30j) · `full`
- `format` : `json` (défaut) · `html`
- `sensor_id` (optionnel) : filtre sur un seul capteur

```
GET /v1/report?type=weekly&format=html
```

Réponse : document HTML complet (KPIs, planning, tableau capteurs) si `format=html`,
sinon JSON structuré équivalent.

Erreurs : `400` si `type` invalide ; `503` si `reporting_module.py` indisponible.

---

### Alertes

#### `GET /v1/alerts`
Auth : non · Rate limit : 30/min

Historique des alertes externes envoyées (email/webhook/SMS) par `AlertManager`, avec le
statut de livraison par canal (booléens uniquement — jamais d'adresse, d'URL ou
d'identifiant exposé). Paramètre query : `limit` (défaut 50).

```json
{ "enabled": true, "stats": { "...": "..." }, "alerts": [ { "sensor_id": "8f7f2f7e", "risk_level": "CRITIQUE", "channels_sent": { "email": true, "webhook": false, "sms": false }, "...": "..." } ] }
```

Si `alert_config.json` n'est pas configuré : `{ "enabled": false, "alerts": [] }`.

#### `GET /v1/alerts/stats`
Auth : non · Rate limit : 30/min

Statistiques globales du gestionnaire d'alertes : total envoyées, répartition par
niveau, cooldowns actifs.

---

### Pipeline (ré-entraînement)

Workflow complet de ré-entraînement à partir d'un dump SQL MariaDB (`ai_cp`), exécuté en
arrière-plan (thread) : parsing SQL → CSV, entraînement des 6 modèles, rechargement à
chaud dans l'API.

#### `GET /pipeline`
Auth : non · Rate limit : aucun
Sert la page HTML d'upload (`pipeline_upload.html`). Exclue du schéma OpenAPI
(`include_in_schema=False`).

#### `POST /v1/pipeline/upload`
Auth : **admin uniquement** (`require_admin_key` — une clé opérateur reçoit **403**) · Rate limit : 5/min

Reçoit un fichier `.sql`, le sauvegarde en flux (blocs de 4 Mo, pour éviter de charger
un dump de plusieurs centaines de Mo entièrement en mémoire), puis démarre en tâche de
fond :
1. Parsing SQL → CSV (`generate_dataset_from_sql.py`)
2. Entraînement des modèles (`train_model_v3_unsupervised.py`)
3. Rechargement à chaud des modèles dans l'API (aucun redémarrage requis)

**Requête** : `multipart/form-data`
- `file` : fichier `.sql` (requis)
- `train_mode` : `"full"` ou `"fast"` (défaut `"full"`)

```bash
curl -X POST http://localhost:8000/v1/pipeline/upload \
  -H "X-API-Key: <clé_admin>" \
  -F "file=@dump_ai_cp.sql" \
  -F "train_mode=full"
```

**Réponse** (202-like, statut 200) :
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "running",
  "filename": "dump_ai_cp.sql",
  "size_mb": 87.4,
  "poll_url": "/v1/pipeline/status/a1b2c3d4e5f6"
}
```

Erreurs : `400` si l'extension n'est pas `.sql` ; `401`/`403` selon le rôle de la clé.

#### `GET /v1/pipeline/status/{job_id}`
Auth : **oui** · Rate limit : 60/min

Statut d'un job en cours — à interroger toutes les 2-3 secondes (polling). Paramètre
query `since` (défaut 0) : n'retourne que les logs postérieurs à cet index, pour éviter
de retransmettre tout l'historique à chaque appel.

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "running",
  "step": 4,
  "step_name": "Entraînement des modèles ML...",
  "progress": 58,
  "elapsed": "1m 12s",
  "new_logs": [ { "text": "✅ LOF entraîné", "type": "s" } ],
  "total_logs": 34,
  "results": null,
  "created_at": "2026-08-07T09:55:00"
}
```
`status` : `running` · `done` · `error`. `results` (non `null` une fois terminé) inclut
`auc`, `f1`, `n_measures`, `n_anomalies`, `top_anomalies`.

Erreurs : `404` si `job_id` inconnu.

#### `GET /v1/pipeline/jobs`
Auth : **oui** · Rate limit : 30/min

Liste des 10 derniers jobs de pipeline (résumé : `job_id`, `status`, `filename`,
`progress`, `elapsed`, `created_at`).

---

## 7. Tableau récapitulatif

| Méthode | Endpoint | Auth | Rate limit (req/min) | Tag |
|---|---|---|---|---|
| GET | `/` | Non | — | Système |
| GET | `/health` | Non | — | Système |
| GET | `/metrics` | Non | 30 | Système |
| GET | `/v1/model-card` | Non | 30 | Système |
| GET | `/v1/system-limits` | Oui | 30 | Système |
| POST | `/v1/predict` | Oui | 60 | IA / Prédiction |
| POST | `/v1/predict-rul` | Oui | 60 | IA / Prédiction |
| POST | `/v1/iot-predict` | Oui | 60 | IA / Prédiction |
| POST | `/v1/spectral-analysis` | Oui | 20 | IA / Prédiction |
| GET | `/v1/spectral/{sensor_id}` | Non | 30 | IA / Prédiction |
| GET | `/v1/health-score/{sensor_id}` | Non | 60 | IA / Prédiction |
| GET | `/v1/history/{sensor_id}` | Non | 60 | IA / Prédiction |
| GET | `/v1/alert-level/{sensor_id}` | Non | 60 | IA / Prédiction |
| GET | `/v1/results` | Non | 60 | IA / Prédiction |
| GET | `/sensors` | Non | 60 | Données |
| GET | `/anomalies` | Non | 60 | Données |
| GET | `/v1/report` | Oui | 20 | Reporting |
| GET | `/v1/alerts` | Non | 30 | Alertes |
| GET | `/v1/alerts/stats` | Non | 30 | Alertes |
| GET | `/pipeline` | Non | — | Pipeline |
| POST | `/v1/pipeline/upload` | **Admin uniquement** | 5 | Pipeline |
| GET | `/v1/pipeline/status/{job_id}` | Oui | 60 | Pipeline |
| GET | `/v1/pipeline/jobs` | Oui | 30 | Pipeline |

> **Note** : `/v1/alerts` et `/v1/alerts/stats` ne portent pas de dépendance
> `require_api_key` dans le code actuel (`api_unified_pythagore.py`), contrairement à ce
> qu'indiquait une version antérieure de la documentation générale — ils sont donc
> accessibles sans clé, comme les autres endpoints de monitoring en lecture seule.
