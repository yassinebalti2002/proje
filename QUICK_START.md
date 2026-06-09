# ⚡ QUICK START — 3 Commandes pour Démarrer

**Durée** : 5 minutes max  
**Pour** : Machine collègue  

---

## 📦 Étape 1 : Copier les Fichiers

Copier le dossier `PROJET_FINAL` complet sur la machine du collègue :
```
PROJET_FINAL/
├── api_unified_pythagore.py
├── realtime_mariadb.py
├── requirements.txt
├── models/              ⭐ CRUCIAL
│   ├── model_*.pkl
│   ├── scaler_v3.pkl
│   ├── pca_v3.pkl
│   └── features_v3.pkl
└── data/                (optionnel)
```

---

## 🐍 Étape 2 : Installer

### Windows PowerShell
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Étape 3 : Lancer

### Option A : Test Rapide (Replay — pas de capteurs)
```bash
python realtime_mariadb.py --replay 50
```

### Option B : Temps Réel (Données capteurs)
```bash
# Terminal 1 : API
python api_unified_pythagore.py

# Terminal 2 : Moteur
python realtime_mariadb.py
```

### Option C : Script Automatique
```bash
# Windows
demarrer_systeme.bat

# Linux/macOS
bash demarrer_systeme.sh
chmod +x demarrer_systeme.sh  # Rendre exécutable
```

---

## ✅ Vérifier que ça Marche

```bash
# Diagnostic
python realtime_mariadb.py --diagnostic

# Doit afficher :
# ✅ mysql-connector-python installé
# ✅ Connexion établie
# ✅ Table trouvée — X lignes
# ✅ API OK
```

---

## 📊 Voir les Résultats

- **Prédictions** : `realtime_results.json` (mis à jour en temps réel)
- **Logs** : `realtime_mariadb.log` (continu)
- **Dashboard** : (Optionnel) `dashboard_realtime.html`

---

## 🔧 Configurer Connexion MariaDB (si différent)

```bash
# Option 1 : Arguments CLI
python realtime_mariadb.py --host 192.168.1.50 --user root --password mon_pass

# Option 2 : Variables d'environnement
$env:MARIADB_HOST = "192.168.1.50"
$env:MARIADB_USER = "root"
python realtime_mariadb.py

# Option 3 : Éditer realtime_mariadb.py ligne 54
DEFAULT_CONFIG = {
    "host": "192.168.1.50",
    "user": "root",
    "password": "mot_de_passe",
}
```

---

## 🆘 Erreur Connexion MariaDB ?

```bash
# 1. Vérifier le serveur est accessible
ping 192.168.1.50

# 2. Relancer diagnostic
python realtime_mariadb.py --diagnostic

# 3. Vérifier identifiants
# → Demander au collègue IT du serveur IoT
```

---

## 📞 Support

| Prob | Solution |
|------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `Connection refused` | `--replay 50` pour test sans réseau |
| `API injoignable` | Lancer `python api_unified_pythagore.py` d'abord |

---

**Besoin d'aide ?** → Voir `GUIDE_INSTALLATION_COLLEGUE.md` (complet)
