# Guide de test — Maintenance Prédictive

Ce guide couvre tous les niveaux de test du projet, du plus rapide (secondes) au plus complet
(système réel avec MariaDB).

---

## 0. Prérequis

```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

Copier `.env.example` en `.env` et renseigner au minimum `API_KEYS` :

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # génère une clé, à coller dans API_KEYS=
```

---

## 1. Suite de tests automatisée (recommandé — 2 à 3 min)

```bash
python run_tests.py --with-server --api-key <ta_clé_API_KEYS>
```

Ce script enchaîne automatiquement :

| Étape | Fichier | Contenu | Serveur requis |
|---|---|---|---|
| 1 | `tests/test_ml_functions.py` | 36 tests unitaires : `safe_mean/std/rms/trend`, `norm01`, `vib_total_pythagorean`, `extract_features`, comparaison de clé API temps constant | Non |
| 2 | `tests/test_security.py` | 18 tests : endpoints publics, authentification 401/200, rate limiting 429, CORS | Oui |
| 3 | `tests/test_api_final.py` | 8 tests : `/`, `/health`, `/metrics`, `/sensors`, `/anomalies`, `/v1/predict`, `/v1/predict-rul`, `/v1/health-score` | Oui |

**Piège connu** : `run_tests.py` lit `API_KEYS` via `os.getenv()`, qui ne charge **pas** automatiquement `.env`
(pas de `python-dotenv` dans ce script). Sans `--api-key` explicite, les étapes 2 et 3 échouent en 401.
Toujours passer `--api-key`.

Sans `--with-server`, seule l'étape 1 (unitaire) tourne — pratique pour un check rapide sans rien démarrer.

### Lancer un seul fichier de tests

```bash
pytest tests/test_ml_functions.py -v                     # pas de serveur requis
pytest tests/test_security.py --api-key <clé> -v         # serveur doit déjà tourner
pytest tests/test_api_final.py --api-key <clé> -v
```

---

## 2. Lancer l'API seule (sans MariaDB)

L'API elle-même **ne dépend pas de MariaDB** au démarrage — seul le moteur temps réel
(`realtime_mariadb.py`) s'y connecte. Utile pour tester les endpoints `/v1/predict`,
`/v1/predict-rul`, `/v1/iot-predict` directement.

```bash
python api_unified_pythagore.py
```

- Swagger interactif : http://localhost:8000/docs — cliquer **Authorize**, coller la clé API
  (celle de `.env` → `API_KEYS=`), puis tester chaque endpoint directement dans le navigateur.
- Health check : `curl http://localhost:8000/health`

---

## 3. Test réel complet (API + MariaDB + dashboard)

### 3.a Le plus simple : `LANCER.bat` (Windows)

```
LANCER.bat
```

Menu interactif : choisir `1 - REPLAY SQL` pour rejouer des données historiques réelles depuis
MariaDB, ou `2 - TEMPS RÉEL` pour lire les capteurs IFM en direct. Le script lit `.env`
automatiquement (host/user/password/database) et ouvre 3 fenêtres (API, moteur, dashboard).

**Bug corrigé (juillet 2026)** : des guillemets imbriqués dans les commandes `start ... cmd /c "..."`
faisaient échouer silencieusement le démarrage de l'API et du moteur (`La syntaxe du nom de
fichier, de répertoire ou de volume est incorrecte.`) — le dashboard affichait alors "Connexion
impossible" en boucle. Corrigé en retirant le `set "API_KEYS=..."` redondant à l'intérieur de la
commande déjà entre guillemets (le process enfant hérite de toute façon de la variable depuis le
script parent).

### 3.b Manuellement (3 terminaux)

```bash
# Terminal 1 — API
python api_unified_pythagore.py

# Terminal 2 — Moteur temps réel (vraies données MariaDB)
python realtime_mariadb.py --host localhost --user root --password <mdp> --database ai_cp \
    --replay 500 --window 5 --poll 1 --timeout 30

# Terminal 3 — Dashboard
python -m http.server 3000
```

Puis ouvrir http://localhost:3000/dashboard_predictive.html

**Recommandations apprises en testant** :
- `--timeout 30` (au lieu du défaut 10s) : évite les timeouts HTTP quand plusieurs capteurs
  remplissent leur fenêtre glissante en même temps lors d'un `--replay` volumineux.
- `--replay` avec au moins **500 lignes** : en dessous, les fenêtres de N mesures par capteur
  ne se remplissent pas assez pour déclencher de vraies prédictions (une mesure physique =
  plusieurs lignes SQL, une par grandeur : température, vibration X/Y/Z...).
- Vérifier que les prédictions arrivent bien : `cat realtime_results.json | python -m json.tool`

### 3.c ⚠️ Ne pas utiliser `realtime_simulator.py` pour tester l'API réelle

Ce script génère des données **synthétiques** (pas de vraies mesures), à ne pas confondre avec
un test en conditions réelles. Utile uniquement pour un smoke-test visuel du dashboard sans base
de données.

---

## 4. Docker

```bash
cp .env.example .env   # remplir les valeurs
docker compose up
```

| Service | URL |
|---|---|
| API | http://localhost:8000/docs |
| Dashboard | http://localhost:3000/dashboard_predictive.html |

---

## 5. Ré-entraînement des modèles (si tu modifies le pipeline ML)

⚠️ Ces scripts **écrasent** les fichiers dans `models/`. Une sauvegarde des modèles avant
correction du data leakage a été faite dans `models_backup_before_leakage_fix/` — à consulter
si besoin de comparer ou de revenir en arrière.

```bash
# 6 modèles de détection (IF, LOF, OCSVM, ECOD, HBOS, COPOD)
# Validation croisée par capteur (GroupKFold 3 folds) + modèle final sur toutes les données
# — voir DOCUMENTATION.md § Pipeline ML
python train_model_v3_unsupervised.py --db-host localhost --db-user root --db-pass <mdp> --db-name ai_cp

# Modèle RUL (GradientBoostingRegressor, split par moteur)
python train_rul_model.py
```

Durée observée sur cette machine : **~5-6 min** pour les 6 modèles + CV (un premier essai avait
pris ~11h à cause d'une contention système ponctuelle, non reproduite depuis), **~1 min** pour le
modèle RUL.

Après ré-entraînement, relancer la suite complète de tests (§1) pour vérifier que l'API charge
bien les nouveaux artefacts et que rien n'est cassé.

---

## 6. Ce qui n'est PAS testable dans un environnement de développement classique

- **Capteurs IFM réels / gateway réseau** (`realtime_ifm_direct.py`, `gateway_ifm_simulator.py`) :
  nécessite le matériel physique ou un simulateur réseau dédié.
- **Déploiement Render** : nécessite un push GitHub + compte Render configuré.
- **Alertes email/SMS/Slack réelles** (`alert_manager.py`) : nécessite de remplir
  `alert_config.json` (jamais commiter ce fichier une fois rempli — voir `.gitignore`).

---

## Checklist rapide avant de considérer le projet "prêt"

- [ ] `python run_tests.py --with-server --api-key <clé>` → 62/62 tests passent
- [ ] `curl http://localhost:8000/health` → `models_loaded: true`, 6 modèles listés
- [ ] Dashboard accessible et affiche des données après un replay MariaDB réel
- [ ] `.env` rempli, jamais commité (`git status` ne doit pas le montrer)
