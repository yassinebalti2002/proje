# Runbook d'exploitation — Maintenance Prédictive IA

Document opérationnel : que faire quand quelque chose ne va pas. À adapter avec
les vrais contacts/astreintes de votre organisation avant usage réel — les
champs `<À COMPLÉTER>` sont volontairement laissés vides.

---

## 1. Contacts et astreinte

| Rôle | Nom | Contact | Disponibilité |
|---|---|---|---|
| Responsable technique | <À COMPLÉTER> | <À COMPLÉTER> | <À COMPLÉTER> |
| Astreinte niveau 1 (ops) | <À COMPLÉTER> | <À COMPLÉTER> | <À COMPLÉTER> |
| Astreinte niveau 2 (dev/data science) | <À COMPLÉTER> | <À COMPLÉTER> | <À COMPLÉTER> |
| Contact client / usine | <À COMPLÉTER> | <À COMPLÉTER> | <À COMPLÉTER> |

**Règle d'escalade** : si un incident n'est pas résolu sous <À COMPLÉTER : ex. 30 min>,
escalader au niveau supérieur.

---

## 2. Vérification rapide de l'état du système

Commande unique de diagnostic :
```
python health_watchdog.py --once
```
Vérifie l'API et MariaDB en un seul appel, sans lancer la surveillance continue.

Endpoints de diagnostic :
| Vérifier | Commande |
|---|---|
| API en vie | `curl http://localhost:8000/health` |
| Modèles chargés | Champ `models_loaded` dans la réponse ci-dessus |
| Métriques modèle actuel | `curl http://localhost:8000/metrics` (nécessite une clé) |
| Traçabilité complète (model card) | `curl http://localhost:8000/v1/model-card` |
| Logs API | `type logs_api.txt` |
| Logs moteur temps réel | `type logs_moteur.txt` |
| Journal d'audit (qui a fait quoi) | `type audit.log` |
| Journal du watchdog | `type watchdog.log` |

---

## 3. Incidents courants et procédure de résolution

### 3.1 API injoignable (health check échoue)

**Symptôme** : `curl http://localhost:8000/health` timeout ou erreur de connexion.
Le watchdog envoie une alerte "API injoignable" après 3 échecs consécutifs (~90s
par défaut).

**Diagnostic** :
1. Le processus tourne-t-il ? `Get-Process python` (Windows) / `ps aux | grep api_unified`
2. Consulter `logs_api.txt` — chercher une exception Python en fin de fichier
3. Le port 8000 est-il déjà utilisé par un autre processus ?

**Résolution** :
- Process mort → relancer : `python api_unified_pythagore.py` (ou `.\LANCER.bat`)
- Modèles manquants (`models/*.pkl` absents) → vérifier que le dossier `models/` n'a
  pas été supprimé/déplacé par erreur ; restaurer depuis une sauvegarde si besoin
- Crash récurrent après relance → escalader niveau 2, ne pas boucler indéfiniment
  sur des relances automatiques sans investiguer la cause

### 3.2 MariaDB injoignable

**Symptôme** : moteur temps réel (`logs_moteur.txt`) affiche des erreurs de connexion
répétées ; watchdog alerte "MariaDB injoignable".

**Diagnostic** :
1. Le service MariaDB tourne-t-il sur le serveur hôte ?
2. Le réseau entre le serveur applicatif et le serveur de base de données est-il
   opérationnel (`ping`, `telnet <host> 3306`) ?
3. Les identifiants dans `.env` sont-ils toujours valides (mot de passe changé ?) ?

**Résolution** :
- Service arrêté → le redémarrer (`net start MariaDB` sous Windows si configuré en
  service, ou `systemctl start mariadb` sous Linux)
- Le moteur temps réel se reconnecte automatiquement une fois la base disponible —
  vérifier qu'aucune mesure n'a été perdue pendant la coupure (limite connue,
  voir DOCUMENTATION.md)

### 3.3 Beaucoup de fausses alertes remontées par les techniciens

**Symptôme** : plaintes répétées "l'alerte dit CRITIQUE mais le moteur va bien".

**Diagnostic** :
1. Consulter `GET /v1/model-card` pour rappeler le contexte : précision actuelle
   du modèle (voir champ `evaluation.precision`)
2. Vérifier si le taux de fausses alertes a réellement augmenté (comparer
   `alert_history.json` sur plusieurs semaines) ou si c'est un biais de perception

**Résolution** :
- Si le taux a réellement dérivé : envisager un ré-entraînement (`POST
  /v1/pipeline/upload`, réservé aux clés admin) sur des données plus récentes
- Si c'est un compromis assumé (précision volontairement sacrifiée pour le rappel,
  voir DOCUMENTATION_TECHNIQUE.pdf §4.5) : réexpliquer le choix au client, ou
  ajuster le seuil de décision si le contexte métier le justifie (implique de
  reconstruire `threshold_v3.pkl` — action réservée niveau 2)

### 3.4 Espace disque saturé

**Symptôme** : écritures qui échouent, `logs_api.txt` très volumineux.

**Diagnostic** : `Get-ChildItem *.log,*.txt | Sort Length -desc` (Windows) pour
identifier les gros fichiers.

**Résolution** :
- Purger/archiver les anciens logs (`logs_api.txt`, `logs_moteur.txt`,
  `watchdog.log`, `audit.log`) — pas de rotation automatique actuellement, à
  mettre en place si le volume devient un problème récurrent
- `realtime_results.json` et `anomaly_history_persist.json` sont auto-bornés
  (50 entrées/capteur) et ne devraient pas croître indéfiniment

### 3.5 Clé API compromise / à révoquer

**Procédure** :
1. Retirer la clé de la variable `API_KEYS` ou `API_KEYS_OPERATOR` dans `.env`
2. Redémarrer l'API (ou appeler le rechargement à chaud si exposé)
3. Vérifier dans `audit.log` l'historique d'utilisation de cette clé (recherche
   par empreinte, jamais la clé en clair n'y figure)
4. Générer et distribuer une nouvelle clé : `python -c "import secrets; print(secrets.token_hex(32))"`

---

## 4. Ce qui N'EST PAS couvert par ce système (à ne jamais oublier)

- **Pas de redondance** : une seule instance API, une seule base de données. Une
  panne serveur coupe la surveillance jusqu'à intervention manuelle.
- **RUL non validé sur pannes réelles** — voir `/v1/model-card`. Toute décision
  d'arrêt machine doit être confirmée par un technicien, jamais automatisée
  uniquement sur la base du RUL affiché.
- **Pas de sauvegarde automatique de la base MariaDB** — à mettre en place
  séparément selon la politique de sauvegarde de l'entreprise.

---

## 5. Historique des incidents

| Date | Incident | Cause | Résolution | Durée |
|---|---|---|---|---|
| <À COMPLÉTER> | | | | |
