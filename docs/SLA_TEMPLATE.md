# Modèle d'accord de niveau de service (SLA) — Maintenance Prédictive IA

⚠️ **Ceci est un modèle de document, pas un SLA signé ou juridiquement valide.**
Il doit être revu par un juriste et adapté avant toute signature avec un client.
Les valeurs entre `<>` sont des exemples à négocier, pas des engagements actuels.

---

## 1. Objet

Le présent document définit les engagements de service pour le système de
maintenance prédictive IA fourni par <FOURNISSEUR> à <CLIENT>, couvrant la
disponibilité, le support, et les limites de responsabilité.

## 2. Définitions

- **Disponibilité** : période durant laquelle l'API répond avec succès à
  `GET /health`.
- **Incident critique (P1)** : système complètement indisponible, ou fournissant
  des données manifestement erronées affectant la sécurité.
- **Incident majeur (P2)** : dégradation significative (ex. fausses alertes en
  hausse notable, latence anormale) sans indisponibilité totale.
- **Incident mineur (P3)** : anomalie n'affectant pas l'usage principal.

## 3. Engagements de disponibilité (exemple à négocier)

| Niveau de service | Disponibilité cible | Fenêtre de mesure |
|---|---|---|
| <Standard> | <99,0 %> | Mensuelle, hors maintenance planifiée |
| <Premium> | <99,5 %> | Mensuelle, hors maintenance planifiée |

**Non couvert par le calcul de disponibilité** :
- Fenêtres de maintenance planifiée, annoncées ≥ <48h> à l'avance
- Pannes dues à l'infrastructure du client (réseau usine, alimentation électrique,
  capteurs physiques défaillants)
- Cas de force majeure

## 4. Temps de réponse par sévérité (exemple à négocier)

| Sévérité | Temps de première réponse | Temps de résolution cible |
|---|---|---|
| P1 — Critique | <1h> (heures ouvrées) / <4h> (hors heures ouvrées) | <8h> |
| P2 — Majeur | <4h> (heures ouvrées) | <2 jours ouvrés> |
| P3 — Mineur | <2 jours ouvrés> | <10 jours ouvrés> |

**Heures ouvrées de référence** : <À COMPLÉTER, ex. Lun-Ven 8h-18h>
**Canal de déclaration d'incident** : <À COMPLÉTER : email/téléphone/portail>

## 5. Périmètre du support

**Inclus :**
- Assistance au diagnostic des incidents listés dans `RUNBOOK_INCIDENTS.md`
- Application de correctifs sur le code livré
- Réponse aux questions d'utilisation de l'API

**Exclu (sauf accord contraire) :**
- Développement de nouvelles fonctionnalités
- Support de l'infrastructure réseau/matérielle du client (capteurs, passerelle
  IFM, câblage, alimentation électrique)
- Ré-entraînement périodique des modèles (à contractualiser séparément si souhaité)
- Formation des utilisateurs finaux (à contractualiser séparément si souhaité)

## 6. ⚠️ Limites de responsabilité — section critique à faire valider par un juriste

Le système de détection d'anomalies et d'estimation de durée de vie résiduelle
(RUL) fournit des **estimations statistiques**, pas des garanties de détection :

- Le modèle de détection d'anomalies atteint, sur données réelles, une précision
  d'environ <valeur mesurée — voir `/v1/model-card`> et un rappel d'environ
  <valeur mesurée>. **Il ne détecte pas 100 % des défaillances réelles.**
- Le modèle RUL est partiellement entraîné sur des **données synthétiques**
  (aucune panne réelle confirmée disponible à ce jour — voir `/v1/model-card`,
  champ `rul_estimation.data_provenance`). **Les estimations d'heures avant
  panne ne doivent jamais être la seule base d'une décision d'arrêt machine.**
- <FOURNISSEUR> ne saurait être tenu responsable des dommages directs ou
  indirects résultant d'une panne non détectée ou d'une fausse alerte, dans
  la limite prévue par le contrat commercial associé.

**Toute décision de maintenance ou d'arrêt machine doit être validée par un
technicien qualifié**, le système étant un outil d'aide à la décision et non
un système de décision autonome.

## 7. Révision du présent document

| Version | Date | Auteur | Modification |
|---|---|---|---|
| 0.1 (modèle) | <À COMPLÉTER> | <À COMPLÉTER> | Version initiale |
