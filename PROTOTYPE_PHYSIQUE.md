# Prototype Physique — Banc d'Essai Moteur

## Présentation

Le prototype physique est un banc d'essai compact pour la surveillance de la santé de roulements industriels en temps réel. Il combine un moteur électrique, des capteurs IFM IO-Link, une passerelle IoT et un ordinateur embarqué (PC ou Raspberry Pi 4).

---

## Architecture Matérielle

```
┌─────────────────────────────────────────────────────────────┐
│                    BANC D'ESSAI                             │
│                                                             │
│  ┌──────────┐    IO-Link    ┌──────────────────────────┐   │
│  │ Capteur  │──────────────▶│                          │   │
│  │IFM VVB001│               │  Passerelle IFM          │   │
│  │(Vib+Temp)│               │  AL1352 / AL1322         │   │
│  └──────────┘               │  IP : 192.168.1.50       │   │
│                              │  Port HTTP : 80          │   │
│  ┌──────────┐    IO-Link    │                          │   │
│  │ Capteur  │──────────────▶│  19 ports IO-Link        │   │
│  │IFM VSE002│               │  + 4 ports I/O           │   │
│  │(Vib 3D)  │               └──────────────┬───────────┘   │
│  └──────────┘                              │ Ethernet       │
│                                            │                │
│  ┌──────────┐    4-20mA     ┌─────────────▼─────────────┐  │
│  │ Capteur  │──────────────▶│   PC / Raspberry Pi 4     │  │
│  │ Courant  │               │   Python 3.11              │  │
│  │(optionnel│               │   MariaDB + FastAPI        │  │
│  └──────────┘               │   Port 8000 (API)          │  │
│                              └───────────────────────────┘  │
│  ┌──────────────────────────────────────┐                   │
│  │  MOTEUR ÉLECTRIQUE                   │                   │
│  │  Type    : Asynchrone triphasé       │                   │
│  │  Puissance: 0.37 – 1.5 kW           │                   │
│  │  Vitesse  : 1400 – 1450 tr/min      │                   │
│  │  Roulement: SKF 6205-2RS (côté NDE) │                   │
│  │  Charge   : Frein magnétique / arbre│                   │
│  └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Liste des Composants (Bill of Materials)

### Capteurs et Instruments

| Réf. | Composant | Modèle | Qté | Description |
|------|-----------|--------|-----|-------------|
| S1   | Capteur vibration + température | IFM VVB001 | 1–4 | IO-Link, mesure 3 axes (X/Y/Z), plage ±8g, 0-85°C |
| S2   | Capteur vibration avancé | IFM VSE002 | 1–2 | IO-Link, RMS + Peak-to-Peak + facteur de crête |
| S3   | Capteur courant (optionnel) | IFM SI6000 ou pince de Rogowski | 1 | Plage 0-30A (4-20mA) |
| S4   | Capteur température (optionnel) | PT100 / IFM TN-series | 1 | Contrôle température boîtier |

### Passerelle IoT

| Réf. | Composant | Modèle | Qté | Description |
|------|-----------|--------|-----|-------------|
| GW1  | Passerelle IO-Link | IFM AL1352 | 1 | 8 ports IO-Link + Ethernet, protocole HTTP REST |
| GW1b | Alternative passerelle | IFM AL1322 | 1 | 4 ports IO-Link, version économique |

### Ordinateur de Traitement

| Réf. | Composant | Modèle | Qté | Description |
|------|-----------|--------|-----|-------------|
| PC1  | Ordinateur embarqué | Raspberry Pi 4 (4GB RAM) | 1 | Déploiement Edge, ARM64 |
| PC1b | Alternative PC | Mini-PC Intel NUC | 1 | Performances supérieures |
| SD1  | Carte microSD | Classe A2, 64GB | 1 | OS + base de données |
| PS1  | Alimentation | 5V/3A USB-C (RPi) | 1 | Alimentation Raspberry Pi |

### Moteur et Banc

| Réf. | Composant | Modèle | Qté | Description |
|------|-----------|--------|-----|-------------|
| M1   | Moteur asynchrone | 0.37 kW, 230/400V, 1440 tr/min | 1 | Moteur d'essai principal |
| B1   | Roulement sain | SKF 6205-2RS | 2 | Roulements en bon état |
| B2   | Roulement défectueux | SKF 6205-2RS (usé) | 1 | Pour tests de détection de défauts |
| FR1  | Structure banc | Profilés aluminium 40×40 | 1 set | Bâti mécanique modulaire |
| CP1  | Couplage | Accouplement flexible | 1 | Liaison moteur/charge |
| CH1  | Charge | Frein à poudre magnétique | 1 | Simulation charge variable |

### Câblage et Connectique

| Réf. | Composant | Modèle | Qté | Description |
|------|-----------|--------|-----|-------------|
| CB1  | Câble IO-Link | IFM E11898 (M12 5 broches) | 4 | Longueur 2m par capteur |
| CB2  | Câble Ethernet | CAT5e RJ45 | 2 | Passerelle ↔ Switch/PC |
| SW1  | Switch Ethernet | 8 ports 100Mbit | 1 | Réseau local banc d'essai |
| PS2  | Alimentation DIN | 24V DC / 2A | 1 | Alimentation capteurs IO-Link |
| TB1  | Bornier DIN | Wago 2273 | 1 set | Distribution 24V aux capteurs |

---

## Schéma de Câblage

### Connexion Capteur IFM VVB001 → Passerelle AL1352

```
Capteur VVB001                    Passerelle AL1352
(Connecteur M12, 5 broches)       (Port IO-Link, M12)
─────────────────────────         ─────────────────────
Broche 1 (L+)  ──────────────────── L+ (24V DC)
Broche 3 (L-)  ──────────────────── L- (GND)
Broche 4 (C/Q) ──────────────────── C/Q (IO-Link)
Broche 2 (n/c) ─── Non connecté
Broche 5 (n/c) ─── Non connecté

Câble : IFM E11898, blindé, longueur max 20m
```

### Connexion Passerelle → Réseau

```
Passerelle AL1352
│
├── Port Ethernet RJ45 ─────────────── Switch 8 ports
│                                         │
│                                         ├── PC/Raspberry Pi 4
│                                         └── (Accès dashboard)
│
└── Alimentation M12 ─── 24V DC / 2A (PS2)
```

### Montage Capteurs sur Moteur

```
           CÔTÉ ACCOUPLEMENT (DE)    CÔTÉ OPPOSÉ (NDE)
                  ┌──────────────────────────┐
Capteur S1 ─────▶│   MOTEUR 0.37 kW         │◀───── Capteur S2
(vib. radiale)    │   1440 tr/min            │       (vib. axiale)
                  │   SKF 6205-2RS           │
                  └──────────────────────────┘
                         │
                    Roulement à
                    surveiller
                    (coté NDE)

Position capteur :
  • Fixation par goujon fileté M5 ou colle cyanoacrylate
  • Direction Z du capteur alignée avec l'axe de rotation
  • Distance : directement sur le carter, au-dessus du roulement
  • Eviter les zones de faible rigidité (couvercles plastique)
```

---

## Configuration Réseau

| Appareil | Adresse IP | Port | Protocole |
|---------|-----------|------|----------|
| Passerelle IFM AL1352 | 192.168.1.50 | 80 | HTTP REST |
| PC / Raspberry Pi | 192.168.1.100 | 8000 | FastAPI |
| Dashboard HTML | 192.168.1.100 | 80 | Nginx |
| MariaDB | 192.168.1.100 | 3306 | MySQL |

**Sous-réseau recommandé :** 192.168.1.0/24  
**Gateway (si accès internet) :** 192.168.1.1

---

## Configuration de la Passerelle IFM AL1352

### Via interface web (http://192.168.1.50)

1. Accéder à l'interface web de la passerelle
2. Dans **Device Settings** → **Network** : configurer l'IP statique 192.168.1.50
3. Dans **IO-Link Master** → Port 1..8 : activer chaque port (mode IO-Link)
4. Dans **Data Storage** : activer le stockage JSON

### Format de données retourné par la passerelle

```json
{
  "SensorNodeId": "8f7f2f7e",
  "MeasDetails": {
    "Id": "a1b2c3d4",
    "Timestamp": "2026-06-04T14:30:00.000Z"
  },
  "gph": "vibration_x",
  "data": {
    "Vibration": {
      "RMS": { "X": 245.3, "Y": 180.1, "Z": 312.7 },
      "Peak": { "X": 680.0, "Y": 520.0, "Z": 890.0 }
    },
    "Temperature": 42.5,
    "Acceleration": {
      "RMS": 312.7,
      "P2P": 1780.0,
      "Z2P": 890.0,
      "Crest": 2.85
    }
  }
}
```

---

## Procédure de Mise en Service

### Étape 1 — Montage mécanique

1. Fixer le moteur sur la structure aluminium avec 4 vis M10
2. Aligner l'arbre moteur avec la charge via comparateur ou règle de précision
3. Monter les roulements SKF 6205-2RS dans les paliers (côté NDE = côté de surveillance)
4. Fixer les capteurs IFM sur le carter moteur (vissage M5 + filetage adapté)
5. Vérifier le serrage et l'alignement des capteurs

### Étape 2 — Câblage électrique

1. **COUPURE ÉLECTRIQUE OBLIGATOIRE avant tout câblage**
2. Alimenter le rail DIN 24V via l'alimentation PS2
3. Câbler les capteurs VVB001/VSE002 vers les ports IO-Link de la passerelle AL1352
4. Vérifier la continuité du câble avec un multimètre (broche L+ = 24V, L- = 0V)
5. Connecter la passerelle au switch Ethernet via câble CAT5e

### Étape 3 — Configuration réseau

```bash
# Sur le Raspberry Pi 4 / PC
sudo nano /etc/dhcpcd.conf
# Ajouter :
# interface eth0
# static ip_address=192.168.1.100/24
# static routers=192.168.1.1

# Tester la connexion à la passerelle
ping 192.168.1.50
curl http://192.168.1.50/iolinkmaster/port[1]/iolinkdevice/pdin
```

### Étape 4 — Démarrage du système logiciel

```bash
# Cloner ou copier le projet sur le Pi
cd /home/pi/PROJET_FINAL_V2

# Lancer en mode production
bash demarrer_systeme.sh

# Ou Docker
docker-compose up -d
```

### Étape 5 — Vérification

1. Ouvrir http://192.168.1.100:8000/docs → vérifier que l'API répond
2. Ouvrir dashboard_realtime.html → vérifier les données temps réel
3. Faire tourner le moteur 5 minutes → vérifier la réception des vibrations
4. Vérifier dans MariaDB : `SELECT COUNT(*) FROM ai_cp.full_data;` doit augmenter

---

## Paramètres d'Acquisition Recommandés

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Fréquence polling | 2 secondes | Compromis réactivité / charge réseau |
| Fenêtre glissante | 10 mesures | ~20s d'historique par capteur |
| Fréquence d'échantillonnage capteur | 400 Hz (VVB001) | Max capteur IFM, BPFI < 200 Hz |
| Filtre anti-repliement | Intégré capteur | Filtre passe-bas à 200 Hz |
| Plage de mesure | ±8g | Adapté roulements industriels 0.37-1.5 kW |
| Résolution | 1 mg | Résolution IFM VVB001 |

---

## Seuils Industriels de Référence (ISO 10816-3)

| Niveau | Vibration RMS (mg) | Action Recommandée |
|--------|-------------------|-------------------|
| Zone A (neuf) | < 280 mg | Fonctionnement normal |
| Zone B (acceptable) | 280 – 710 mg | Surveillance renforcée |
| Zone C (alarme) | 710 – 1120 mg | Planifier maintenance |
| Zone D (danger) | > 1120 mg | Arrêt immédiat |

Seuils température roulement :
- Normal : < 50°C
- Attention : 50–65°C
- Critique : > 65°C (risque défaillance accélérée)

---

## Fréquences Caractéristiques des Défauts (SKF 6205-2RS)

Pour un moteur tournant à 1450 tr/min (24.17 Hz) :

| Défaut | Fréquence | Harmoniques à surveiller |
|--------|-----------|--------------------------|
| Bague extérieure (BPFO) | 90.1 Hz | 90.1 / 180.2 / 270.3 Hz |
| Bague intérieure (BPFI) | 143.7 Hz | 143.7 / 287.4 / 431.1 Hz |
| Bille (BSF) | 59.5 Hz | 59.5 / 119.0 / 178.5 Hz |
| Cage (FTF) | 10.0 Hz | 10.0 / 20.0 / 30.0 Hz |

Ces fréquences sont calculées automatiquement par `signal_processing.py` selon la vitesse réelle du moteur.

---

## Dépannage Matériel

| Symptôme | Cause possible | Solution |
|---------|----------------|----------|
| Capteur non détecté par la passerelle | Câble défectueux ou mauvaise broche | Tester continuité câble, vérifier brochage M12 |
| Valeurs de vibration nulles | Capteur mal fixé | Revérifier le serrage et la surface de contact |
| Températures aberrantes (>100°C) | Interférence électromagnétique | Câble blindé, mise à la terre du châssis |
| Passerelle inaccessible sur le réseau | Conflit IP ou mauvaise configuration | Réinitialiser la passerelle via bouton RESET (10s) |
| Données incohérentes (oscillations) | Résonance structurale du banc | Ajouter masse amortissante ou raidissement du bâti |
| MariaDB ne reçoit pas de données | Réseau local non routable | Vérifier switch et VLAN, désactiver pare-feu |

---

## Coût Estimatif du Banc d'Essai

| Poste | Coût estimé (EUR) |
|-------|-------------------|
| Capteurs IFM VVB001 × 2 | 600 – 800 |
| Passerelle IFM AL1352 | 400 – 600 |
| Moteur asynchrone 0.37 kW | 150 – 300 |
| Raspberry Pi 4 (4GB) + alimentation | 80 – 120 |
| Câbles IO-Link, Ethernet, connectique | 100 – 150 |
| Structure aluminium, fixations | 150 – 250 |
| Alimentation 24V DIN | 60 – 100 |
| **TOTAL ESTIMÉ** | **1 540 – 2 320 EUR** |

---

## Références

- **IFM VVB001** : [Documentation IFM VVB001](https://www.ifm.com/fr/fr/products/sensor-solutions/vibration-sensors/VVB001.html)
- **IFM AL1352** : Manuel technique AL1352 (IO-Link Master)
- **SKF 6205-2RS** : Catalogue SKF roulements rigides à billes
- **ISO 10816-3** : Évaluation des vibrations mécaniques par mesure sur les parties non tournantes — Machines industrielles > 15 kW
- **ISO 20816-3** : Norme mise à jour pour surveillance des vibrations
- **IEC 61131-9** : IO-Link — Interface de communication pour capteurs/actionneurs
