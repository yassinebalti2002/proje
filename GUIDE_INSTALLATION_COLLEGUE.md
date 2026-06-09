# 🚀 Guide Installation — Machine Collègue (Test Temps Réel)

**Date** : May 20, 2026  
**Objectif** : Tester le système complet avec données temps réel depuis MariaDB IoT  
**Durée estimée** : 15–20 minutes

---

## 📋 Checklist Pré-Installation

- [ ] Python 3.10+ installé sur la machine du collègue
- [ ] Accès réseau local au serveur MariaDB (`192.168.1.50:3306`)
- [ ] Permission SSH/RDP à la machine du collègue
- [ ] Zip du projet PROJET_FINAL complet

---

## 🔧 Étape 1 : Transfert des Fichiers

### Option A : Par USB/Partage Réseau
```bash
📦 STRUCTURE À COPIER SUR LA MACHINE DU COLLÈGUE :
├── api_unified_pythagore.py          (API FastAPI)
├── realtime_mariadb.py                (Moteur temps réel)
├── api_client.py                      (Client HTTP — optionnel)
├── realtime_results.json              (Résultats — créé auto)
├── requirements.txt                   (Dépendances)
├── models/                            (Dossier ML — CRUCIAL)
│   ├── model_if_v3.pkl
│   ├── model_lof_v3.pkl
│   ├── model_ocsvm_v3.pkl
│   ├── model_ecod_v3.pkl
│   ├── scaler_v3.pkl
│   ├── pca_v3.pkl
│   ├── features_v3.pkl
│   └── metrics_v3.csv
├── data/                              (Dataset — optionnel)
│   └── dataset_2026_with_acc.csv
└── dashboard_realtime.html            (Affichage web — optionnel)
```

### Option B : Cloner depuis Git
```bash
# Si vous avez un dépôt Git partagé
git clone <url_repo> PROJET_FINAL
cd PROJET_FINAL
```

---

## 🐍 Étape 2 : Installation Environnement Python

### 2.1 Créer un Virtualenv
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2.2 Installer les Dépendances
```bash
pip install -r requirements.txt
```

**Vérification** :
```bash
pip list | grep fastapi
# Doit afficher : fastapi       0.115.0
```

---

## 🔌 Étape 3 : Configuration MariaDB

### 3.1 Paramètres de Connexion
Avant de lancer, **configurer les accès** :

**Fichier** : `realtime_mariadb.py` (lignes 54–62)

```python
DEFAULT_CONFIG = {
    "host":     "192.168.1.50",              # ✅ IP serveur IoT (à ajuster si différent)
    "port":     3306,                        # ✅ Port MariaDB
    "user":     "root",                      # ✅ Utilisateur DB
    "password": "ton_mot_de_passe",          # ✅ MOT DE PASSE — À REMPLACER
    "database": "ai_cp",                     # Base de données
    "table":    "full_data",                 # Table capteurs
}
```

**⚠️ IMPORTANT — 3 Façons de Configurer :**

#### Option 1️⃣ : Variables d'Environnement (Recommandé — sécurisé)
```bash
# Windows PowerShell
$env:MARIADB_HOST = "192.168.1.50"
$env:MARIADB_USER = "root"
$env:MARIADB_PASSWORD = "mot_de_passe_securise"
$env:MARIADB_DATABASE = "ai_cp"
$env:MARIADB_TABLE = "full_data"

# Linux/Mac Bash
export MARIADB_HOST="192.168.1.50"
export MARIADB_USER="root"
export MARIADB_PASSWORD="mot_de_passe_securise"
```

#### Option 2️⃣ : Arguments CLI
```bash
python realtime_mariadb.py \
  --host 192.168.1.50 \
  --user root \
  --password ton_mot_de_passe \
  --database ai_cp \
  --table full_data
```

#### Option 3️⃣ : Éditer le code (moins sécurisé)
Modifier `realtime_mariadb.py` ligne 54–62

---

## ✅ Étape 4 : Test de Connectivité

### 4.1 Diagnostic Rapide
```bash
python realtime_mariadb.py --diagnostic
```

**Résultat attendu** :
```
============================================================
  DIAGNOSTIC — MariaDB IoT → API
============================================================

1. Dépendance mysql-connector-python...
   ✅ mysql-connector-python installé

2. Connexion MariaDB 192.168.1.50:3306/ai_cp...
   ✅ Connexion établie

3. Vérification table 'full_data'...
   ✅ Table trouvée — 1,245,600 lignes
   Dernières lignes reçues :
     id=1245599 | capteur=8f7f2f7e | gph=temperature
     id=1245598 | capteur=8f7f2f7d | gph=vibration_y

4. API FastAPI http://localhost:8000...
   ❌ API injoignable : Connection refused
   → Lance d'abord : python api_unified_pythagore.py

✅ Diagnostic OK — Lance maintenant : python api_unified_pythagore.py
```

### 4.2 Troubleshooting Connexion MariaDB

**Erreur** : `Connection refused — timeout`
```bash
# ✓ Tester la route réseau
ping 192.168.1.50

# ✓ Vérifier le port MariaDB
netstat -an | grep 3306

# ✓ Vérifier les credentials (user/password)
# Demander à l'IT du serveur IoT
```

**Erreur** : `Access denied for user 'root'@'machine_collegue'`
```sql
-- Sur le serveur MariaDB (IT IoT)
GRANT ALL ON *.* TO 'root'@'%' IDENTIFIED BY 'mot_de_passe';
FLUSH PRIVILEGES;
```

---

## 🚀 Étape 5 : Lancer le Système Complet

### 5.1 Terminal 1 : API FastAPI

```bash
# Activation venv (si pas déjà fait)
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

# Lancer l'API
python api_unified_pythagore.py
```

**Sortie attendue** :
```
[API] Chargement des modèles V3...
[API] ✅ 4 modèles chargés | Features: 25 | PCA: 5
[API] ✅ Historique restauré depuis anomaly_history_persist.json (0 capteurs)
Uvicorn running on http://0.0.0.0:8000
Press CTRL+C to quit
```

### 5.2 Terminal 2 : Moteur Temps Réel

```bash
# Activation venv
.\venv\Scripts\Activate.ps1

# Lancer le moteur temps réel
python realtime_mariadb.py
```

**Sortie attendue** :
```
[MARIADB] ═══════════════════════════════════════════════════════════════
[MARIADB]   MOTEUR TEMPS RÉEL — MariaDB IoT → API FastAPI
[MARIADB]   MariaDB : 192.168.1.50:3306/ai_cp/full_data
[MARIADB]   API     : http://localhost:8000
[MARIADB]   Fenêtre : 10 mesures | Poll : 2.0s
[MARIADB] ═══════════════════════════════════════════════════════════════

✅ Prêt — en attente de données réelles depuis les capteurs IFM...

Moteur démarré — données réelles depuis MariaDB IoT
Réseau local : 192.168.1.50:3306/ai_cp
```

### 5.3 Terminal 3 : Dashboard (Optionnel)

```bash
# Serveur simple Python pour afficher le dashboard
python -m http.server 8080

# Puis ouvrir navigateur
# http://localhost:8080/dashboard_realtime.html
```

---

## 📊 Modes de Lancement Avancés

### Mode Replay (test sans capteurs)
```bash
# Rejoue les 100 dernières mesures réelles stockées
python realtime_mariadb.py --replay 100
```

### Mode Window Personnalisé
```bash
# Fenêtre de 20 mesures au lieu de 10
python realtime_mariadb.py --window 20 --poll 1.0
```

### Mode Debug
```bash
# Logs détaillés dans realtime_mariadb.log
python realtime_mariadb.py --diagnostic
# Puis vérifier le fichier log
tail -f realtime_mariadb.log
```

---

## 🌐 Étape 6 : Vérifier les Endpoints API

### 6.1 Health Check
```bash
curl http://localhost:8000/health

# Réponse attendue :
{
  "status": "ok",
  "version": "3.1.0",
  "models": ["if", "lof", "ocsvm", "ecod"],
  "features": 25,
  "pca_components": 5
}
```

### 6.2 Voir les Anomalies Détectées
```bash
curl http://localhost:8000/anomalies

# Réponse : liste des anomalies récentes détectées
```

### 6.3 API Docs Interactifs
Ouvrir dans le navigateur :
```
http://localhost:8000/docs
```

---

## 📈 Suivi en Temps Réel

### Fichiers de Résultats

| Fichier | Contenu | Mise à jour |
|---------|---------|-------------|
| `realtime_results.json` | 500 dernières prédictions | Chaque prédiction |
| `realtime_mariadb.log` | Logs détaillés | Continu |
| `anomaly_history_persist.json` | Historique par capteur | Toutes les 5 min |

### Affichage Terminal

Le moteur affiche en continu :
```
────────────────────────────────────────────────────
[14:35:22]  #42  |  Capteur : 8f7f2f7e  |  Source : MariaDB RÉEL
────────────────────────────────────────────────────
🔴 Risque    : MODÉRÉ
🗳️  Votes     : 2/4 modèles
📊 Score     : 0.62
🤖 Modèles   : IF=🟢 | LOF=🔴 | OCSVM=🔴 | ECOD=🟢
⏳ RUL       : 48.5h / 2.02j
💚 Santé     : 62/100
🔔 Alerte    : ATTENTION
💡 Reco      : Surveiller le vibration Z — tendance à la hausse
🌡️  Temp      : 52.3°C
📳 Vib Z     : 245.6 mg
📳 Vib total : 387.6 mg
```

---

## ⚡ Optimisations pour la Machine Collègue

### Si Ressources Limitées (CPU/RAM Faibles)

```bash
# Réduire la fenêtre glissante
python realtime_mariadb.py --window 5 --batch 50 --poll 3.0

# Réduire le nombre de capteurs surveillés
# Éditer realtime_mariadb.py ligne 60 :
MAX_SENSORS = 10  # au lieu de 25
```

### Si Réseau Lent

```bash
# Augmenter le timeout API
python realtime_mariadb.py --timeout 20 --retries 5
```

### Si Données Manquantes

```bash
# Mode replay pour tester sans attendre les capteurs
python realtime_mariadb.py --replay 200 --poll 0.5
```

---

## 🔧 Fichiers de Configuration à Préparer

### `.env` (Optionnel — pour les secrets)
```env
MARIADB_HOST=192.168.1.50
MARIADB_PORT=3306
MARIADB_USER=root
MARIADB_PASSWORD=secure_password_here
MARIADB_DATABASE=ai_cp
MARIADB_TABLE=full_data
```

Puis charger dans Python :
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## ✅ Checklist Final

- [ ] Tous les fichiers copiés (models/ surtout)
- [ ] Python 3.10+ installé
- [ ] `pip install -r requirements.txt` ✅
- [ ] Diagnostic MariaDB ✅ (pas d'erreurs)
- [ ] API FastAPI démarre ✅
- [ ] Moteur temps réel se connecte ✅
- [ ] Prédictions arrivantes dans `realtime_results.json` ✅
- [ ] Dashboard accessible sur http://localhost:8080 ✅

---

## 🆘 Support & Troubleshooting

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError: fastapi` | `pip install -r requirements.txt` |
| `Connection refused — MariaDB` | Ping 192.168.1.50 + vérifier port 3306 |
| `API port 8000 already in use` | `python -m fuser -k 8000/tcp` (Linux) ou changer port |
| `Models not loaded` | Vérifier que `models/` est bien copié |
| `No data flowing` | `--replay 50` pour tester avec données stockées |

---

## 📞 Contact

- **Problème MariaDB** → Contacter IT serveur IoT (192.168.1.50)
- **Problème ML/API** → Consulter logs `realtime_mariadb.log`
- **Problème Réseau** → Vérifier firewall machine + routeur

---

**Prêt à lancer ? 🚀** Commencez par l'Étape 4 (Diagnostic) !
