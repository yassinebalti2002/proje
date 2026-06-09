# ✅ CHECKLIST D'INSTALLATION COLLÈGUE

**Nom du collègue** : ______________________  
**Machine** : ______________________  
**Date** : ______________________  

---

## 📦 PRÉ-INSTALLATION

- [ ] Dossier `PROJET_FINAL` reçu (USB/Réseau/Git)
- [ ] Python 3.10+ installé
  ```bash
  python --version  # Doit afficher 3.10+
  ```
- [ ] Accès réseau au serveur MariaDB (192.168.1.50)
  ```bash
  ping 192.168.1.50  # Doit répondre
  ```

---

## 🐍 INSTALLATION ENVIRONNEMENT

- [ ] Virtualenv créé
  ```bash
  python -m venv venv  # Puis activation
  ```
- [ ] Dépendances installées sans erreurs
  ```bash
  pip install -r requirements.txt
  pip list | grep fastapi  # Doit afficher 0.115.0
  ```

---

## 📂 FICHIERS ESSENTIELS COPIÉS

### Fichiers Python
- [ ] `api_unified_pythagore.py`
- [ ] `realtime_mariadb.py`
- [ ] `requirements.txt`

### Dossier Models (CRUCIAL)
- [ ] `models/model_if_v3.pkl`
- [ ] `models/model_lof_v3.pkl`
- [ ] `models/model_ocsvm_v3.pkl`
- [ ] `models/model_ecod_v3.pkl`
- [ ] `models/scaler_v3.pkl`
- [ ] `models/pca_v3.pkl`
- [ ] `models/features_v3.pkl`
- [ ] `models/metrics_v3.csv`

### Dossier Data (Optionnel)
- [ ] `data/dataset_2026_with_acc.csv`

### Scripts de Démarrage
- [ ] `demarrer_systeme.bat` (Windows)
- [ ] `demarrer_systeme.sh` (Linux/macOS)

### Guides
- [ ] `GUIDE_INSTALLATION_COLLEGUE.md`
- [ ] `QUICK_START.md`

---

## 🔌 CONFIGURATION MARIADB

- [ ] IP/Port MariaDB connu
  - **Host** : 192.168.1.50
  - **Port** : 3306
  - **Database** : ai_cp
  - **Table** : full_data

- [ ] Identifiants MariaDB obtenus
  - **User** : root (ou __________  )
  - **Password** : (**masqué — stocké en variables d'env**)

---

## ✅ TESTS DE CONNECTIVITÉ

### Test 1 : Ping Serveur
```bash
ping 192.168.1.50
```
- [ ] Réponse reçue (pas de timeout)

### Test 2 : Diagnostic
```bash
python realtime_mariadb.py --diagnostic
```
- [ ] ✅ mysql-connector installé
- [ ] ✅ Connexion MariaDB OK
- [ ] ✅ Table trouvée (N lignes)
- [ ] ⚠️  API non lancée (normal à ce stade)

### Test 3 : API Démarrage
```bash
python api_unified_pythagore.py
```
- [ ] ✅ Modèles chargés (4 modèles)
- [ ] ✅ Features: 25 | PCA: 5
- [ ] ✅ Uvicorn running on 0.0.0.0:8000

### Test 4 : Health Check
```bash
curl http://localhost:8000/health
# ou navigateur : http://localhost:8000/health
```
- [ ] Réponse JSON avec status: "ok"

---

## 🚀 LANCEMENT SYSTÈME

### Mode 1 : Replay (Test rapide — pas de capteurs)
```bash
python realtime_mariadb.py --replay 50
```
- [ ] Moteur démarre
- [ ] Prédictions affichées dans le terminal
- [ ] Fichier `realtime_results.json` créé/mis à jour

### Mode 2 : Temps Réel (Données capteurs)

**Terminal 1 — API**
```bash
python api_unified_pythagore.py
```
- [ ] Logs affichés | API ready

**Terminal 2 — Moteur**
```bash
python realtime_mariadb.py
```
- [ ] Moteur prêt | En attente de données

- [ ] Données reçues dans les 30s
  ```
  Mesure reçue — capteur=8f7f2f7e ...
  ```

- [ ] Prédictions affichées
  ```
  ────────────────────────────────
  [14:35:22] #42 | Capteur: 8f7f2f7e
  🔴 Risque: MODÉRÉ
  🗳️ Votes: 2/4 modèles
  ```

---

## 📊 VÉRIFICATION RÉSULTATS

- [ ] Fichier `realtime_results.json` existe
  ```bash
  wc -l realtime_results.json  # Doit avoir plusieurs lignes
  ```

- [ ] Logs `realtime_mariadb.log` générés
  ```bash
  tail -20 realtime_mariadb.log
  ```

- [ ] Historique persisté (`anomaly_history_persist.json`)
  - Créé après 1ère prédiction

---

## 🌐 DASHBOARD (Optionnel)

- [ ] Serveur web local lancé
  ```bash
  python -m http.server 8080
  ```

- [ ] Dashboard accessible
  ```
  http://localhost:8080/dashboard_realtime.html
  ```

- [ ] Affichage des prédictions en temps réel

---

## 🎯 POINTS DE MESURE (KPI)

### Performance Attendue

| Métrique | Valeur Cible | Observé |
|----------|--------------|---------|
| Temps démarrage API | < 5s | ______ |
| Temps 1ère prédiction | < 30s | ______ |
| Prédictions/minute | 2–3 | ______ |
| Fiabilité API | ≥ 90% | ______ |
| Anomalies détectées | 5–15% | ______ |

---

## 🔧 OPTIMISATIONS POSSIBLES

Si ressources limitées :

- [ ] Réduire fenêtre
  ```bash
  python realtime_mariadb.py --window 5
  ```

- [ ] Augmenter intervalle polling
  ```bash
  python realtime_mariadb.py --poll 3.0
  ```

- [ ] Limiter capteurs simultanés
  ```
  # Éditer realtime_mariadb.py
  MAX_SENSORS = 10  # au lieu de 25
  ```

---

## 📝 NOTES / PROBLÈMES RENCONTRÉS

```
________________________________________________________________________

________________________________________________________________________

________________________________________________________________________

________________________________________________________________________
```

---

## 🎓 FORMATION COLLÈGUE

**Documentations à lire (dans l'ordre)** :

1. ⭐ `QUICK_START.md` (3 commandes)
2. 📖 `GUIDE_INSTALLATION_COLLEGUE.md` (complet)
3. 🔍 Code commenté dans `api_unified_pythagore.py`
4. 📊 Dashboard HTML — `dashboard_realtime.html`

**Points clés à expliquer** :

- [ ] Architecture complète (Capteurs → MariaDB → API → Prédictions)
- [ ] 4 modèles ensemble (IF, LOF, OCSVM, ECOD)
- [ ] 25 features utilisées (température, vibrations, courant)
- [ ] Vote majoritaire (2/4 = anomalie)
- [ ] RUL = durée de vie estimée

---

## ✨ SIGNATURE VALIDATION

**Collègue** : ______________________  
**Date** : ______________________  
**Tous les tests passent ?** : ☐ OUI ☐ NON  

**Contact support** :  
📧 Email : ______________________  
📱 Tel : ______________________  
⏰ Disponibilité : ______________________  

---

**Bon test ! 🚀**
