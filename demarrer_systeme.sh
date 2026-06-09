#!/bin/bash
# ╔════════════════════════════════════════════════════════════════════════╗
# ║  DEMARRAGE AUTOMATIQUE — SYSTÈME TEMPS RÉEL                           ║
# ║  Linux / macOS                                                         ║
# ║  Usage : bash demarrer_systeme.sh                                     ║
# ║  ou : chmod +x demarrer_systeme.sh && ./demarrer_systeme.sh          ║
# ╚════════════════════════════════════════════════════════════════════════╝

set -e

clear
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  MOTEUR TEMPS RÉEL — MariaDB IoT → API FastAPI                       ║"
echo "║  Machine : $(hostname)"
echo "║  Date : $(date)"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# ── Vérifier Python ──────────────────────────────────────────────────────
echo "[1/4] Vérification Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé — installez Python 3.10+"
    echo "   Linux : sudo apt install python3 python3-pip"
    echo "   macOS : brew install python3"
    exit 1
fi
echo "✅ Python détecté : $(python3 --version)"
echo ""

# ── Vérifier/Créer virtualenv ────────────────────────────────────────────
echo "[2/4] Préparation virtualenv..."
if [ ! -d "venv" ]; then
    echo "   Création venv..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "✅ Virtualenv activé"
echo ""

# ── Installer dépendances ───────────────────────────────────────────────
echo "[3/4] Installation dépendances..."
pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Échec installation dépendances"
    echo "   Lancez : pip install -r requirements.txt"
    exit 1
fi
echo "✅ Dépendances OK"
echo ""

# ── Diagnostic MariaDB ──────────────────────────────────────────────────
echo "[4/4] Diagnostic MariaDB..."
python realtime_mariadb.py --diagnostic
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Diagnostic échoué — vérifier configuration MariaDB"
    exit 1
fi
echo ""

# ── Choix mode de lancement ─────────────────────────────────────────────
show_menu() {
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║  MODES DE LANCEMENT                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "1 = Temps réel (attend données capteurs)"
    echo "2 = Replay (rejeu 50 dernières mesures stockées)"
    echo "3 = Replay personnalisé (N mesures)"
    echo "4 = Configuration avancée"
    echo "5 = API + Moteur (2 terminaux)"
    echo "6 = Quitter"
    echo ""
}

show_menu
read -p "Choix (1-6) : " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "▶️  Démarrage MODE TEMPS RÉEL..."
        echo "   Données capteurs IFM en direct depuis 192.168.1.50:3306"
        echo ""
        python realtime_mariadb.py
        ;;
    2)
        echo ""
        echo "▶️  Démarrage MODE REPLAY (50 mesures)..."
        echo ""
        python realtime_mariadb.py --replay 50
        ;;
    3)
        read -p "Nombre de mesures à rejouer : " N
        python realtime_mariadb.py --replay $N
        ;;
    4)
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  CONFIGURATION AVANCÉE                                                ║"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        read -p "IP MariaDB [192.168.1.50] : " HOST
        HOST=${HOST:-192.168.1.50}
        
        read -p "Utilisateur [root] : " USER
        USER=${USER:-root}
        
        read -p "Base de données [ai_cp] : " DB
        DB=${DB:-ai_cp}
        
        read -p "Taille fenêtre [10] : " WINDOW
        WINDOW=${WINDOW:-10}
        
        read -p "Intervalle poll [2.0] : " POLL
        POLL=${POLL:-2.0}
        
        echo ""
        echo "▶️  Démarrage avec config..."
        echo "   Host: $HOST | User: $USER | DB: $DB"
        echo "   Window: $WINDOW | Poll: ${POLL}s"
        echo ""
        python realtime_mariadb.py --host $HOST --user $USER --database $DB --window $WINDOW --poll $POLL
        ;;
    5)
        echo ""
        echo "▶️  Démarrage MODE DUAL (API + Moteur)..."
        echo ""
        echo "   Terminal 1 : API FastAPI"
        echo "   Terminal 2 : Moteur temps réel"
        echo ""
        echo "   Ouvrez 2 terminaux et lancez :"
        echo "   1️⃣  source venv/bin/activate && python api_unified_pythagore.py"
        echo "   2️⃣  source venv/bin/activate && python realtime_mariadb.py"
        echo ""
        echo "   Puis attendez 5s et ouvrez le navigateur :"
        echo "   🌐 http://localhost:8000/docs"
        ;;
    6)
        echo ""
        echo "Arrêt — À bientôt! 👋"
        exit 0
        ;;
    *)
        echo "❌ Choix invalide"
        ;;
esac

echo ""
echo "Arrêt — À bientôt! 👋"
