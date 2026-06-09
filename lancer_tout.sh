#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║   LANCEMENT COMPLET EN UNE SEULE COMMANDE                           ║
# ║   Lance en parallèle :                                              ║
# ║     [1] API FastAPI          → http://localhost:8000                ║
# ║     [2] Moteur realtime      → MariaDB IoT (ou --replay N)          ║
# ║     [3] Serveur Dashboard    → http://localhost:3000                ║
# ║                                                                      ║
# ║   Usage :                                                            ║
# ║     bash lancer_tout.sh                   (temps réel)              ║
# ║     bash lancer_tout.sh --replay 100      (replay 100 mesures)      ║
# ║     bash lancer_tout.sh --simulateur      (sans capteurs ni MariaDB)║
# ║                                                                      ║
# ║   Arrêt : Ctrl+C  — arrête TOUT proprement                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── Couleurs ──────────────────────────────────────────────────────────
GREEN="\033[92m"; RED="\033[91m"; YELLOW="\033[93m"
CYAN="\033[96m";  BOLD="\033[1m"; RESET="\033[0m"

clear
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════╗"
echo -e "║   MAINTENANCE PRÉDICTIVE — LANCEMENT COMPLET                 ║"
echo -e "║   $(date '+%d/%m/%Y %H:%M:%S')                                          ║"
echo -e "╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Paramètres ────────────────────────────────────────────────────────
REPLAY=0
SIMULATEUR=0
MOTEUR_ARGS=""

for arg in "$@"; do
  case $arg in
    --replay)   REPLAY="${2}"; shift ;;
    --replay=*) REPLAY="${arg#*=}" ;;
    --simulateur) SIMULATEUR=1 ;;
  esac
done

# ── PIDs des processus enfants ─────────────────────────────────────────
PID_API=0
PID_MOTEUR=0
PID_HTTP=0

# ── Nettoyage à Ctrl+C ─────────────────────────────────────────────────
cleanup() {
    echo ""
    echo -e "${YELLOW}${BOLD}Arrêt demandé — extinction de tous les processus...${RESET}"
    [ $PID_API    -ne 0 ] && kill $PID_API    2>/dev/null && echo -e "  ${RED}✖${RESET} API FastAPI     (PID $PID_API) arrêtée"
    [ $PID_MOTEUR -ne 0 ] && kill $PID_MOTEUR 2>/dev/null && echo -e "  ${RED}✖${RESET} Moteur MariaDB  (PID $PID_MOTEUR) arrêté"
    [ $PID_HTTP   -ne 0 ] && kill $PID_HTTP   2>/dev/null && echo -e "  ${RED}✖${RESET} Dashboard HTTP  (PID $PID_HTTP) arrêté"
    echo ""
    echo -e "${GREEN}Tout arrêté proprement. À bientôt !${RESET}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Étape 1 : Python ───────────────────────────────────────────────────
echo -e "${BOLD}[1/5] Vérification Python...${RESET}"
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo -e "  ${RED}❌ Python non trouvé — installez Python 3.10+${RESET}"
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)
echo -e "  ${GREEN}✅ $($PYTHON --version)${RESET}"

# ── Étape 2 : Virtualenv ───────────────────────────────────────────────
echo ""
echo -e "${BOLD}[2/5] Virtualenv...${RESET}"
if [ ! -d "venv" ]; then
    echo -e "  Création du venv..."
    $PYTHON -m venv venv
fi
source venv/bin/activate
echo -e "  ${GREEN}✅ Virtualenv activé${RESET}"

# ── Étape 3 : Dépendances ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}[3/5] Dépendances...${RESET}"
pip install -q -r requirements.txt
echo -e "  ${GREEN}✅ Dépendances OK${RESET}"

# ── Étape 4 : Vérifier fichiers essentiels ────────────────────────────
echo ""
echo -e "${BOLD}[4/5] Vérification fichiers...${RESET}"
for f in api_unified_pythagore.py realtime_mariadb.py models/model_if_v3.pkl models/features_v3.pkl; do
    if [ ! -f "$f" ]; then
        echo -e "  ${RED}❌ Fichier manquant : $f${RESET}"
        exit 1
    fi
done
echo -e "  ${GREEN}✅ Tous les fichiers présents${RESET}"

# ── Étape 5 : Lancement en parallèle ─────────────────────────────────
echo ""
echo -e "${BOLD}[5/5] Lancement des 3 processus...${RESET}"
echo ""

# --- Processus 1 : API FastAPI ---
echo -e "  ${CYAN}▶ [1/3] API FastAPI sur http://localhost:8000${RESET}"
$PYTHON api_unified_pythagore.py >> logs_api.txt 2>&1 &
PID_API=$!
echo -e "        PID=$PID_API | logs → logs_api.txt"

# Attendre que l'API soit prête (max 15s)
echo -ne "        En attente de l'API"
for i in $(seq 1 15); do
    sleep 1
    echo -n "."
    if curl -s http://localhost:8000/health | grep -q '"status":"ok"' 2>/dev/null; then
        echo -e " ${GREEN}OK !${RESET}"
        break
    fi
    if [ $i -eq 15 ]; then
        echo -e " ${RED}timeout (l'API met du temps, on continue quand même)${RESET}"
    fi
done

# --- Processus 2 : Moteur de données ---
if [ $SIMULATEUR -eq 1 ]; then
    echo -e "  ${CYAN}▶ [2/3] Simulateur (mode sans capteurs)${RESET}"
    $PYTHON realtime_simulator.py --scenario aleatoire >> logs_moteur.txt 2>&1 &
    PID_MOTEUR=$!
    echo -e "        PID=$PID_MOTEUR | logs → logs_moteur.txt"
elif [ "$REPLAY" -ne 0 ] 2>/dev/null; then
    echo -e "  ${CYAN}▶ [2/3] Moteur MariaDB — REPLAY $REPLAY mesures${RESET}"
    $PYTHON realtime_mariadb.py --replay $REPLAY >> logs_moteur.txt 2>&1 &
    PID_MOTEUR=$!
    echo -e "        PID=$PID_MOTEUR | logs → logs_moteur.txt"
else
    echo -e "  ${CYAN}▶ [2/3] Moteur MariaDB — TEMPS RÉEL (192.168.1.50)${RESET}"
    $PYTHON realtime_mariadb.py >> logs_moteur.txt 2>&1 &
    PID_MOTEUR=$!
    echo -e "        PID=$PID_MOTEUR | logs → logs_moteur.txt"
fi

# --- Processus 3 : Serveur HTTP pour le dashboard ---
echo -e "  ${CYAN}▶ [3/3] Dashboard HTTP sur http://localhost:3000${RESET}"
$PYTHON -m http.server 3000 >> logs_http.txt 2>&1 &
PID_HTTP=$!
echo -e "        PID=$PID_HTTP | logs → logs_http.txt"

# ── Résumé final ───────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗"
echo -e "║   ✅ SYSTÈME DÉMARRÉ COMPLÈTEMENT                            ║"
echo -e "╠══════════════════════════════════════════════════════════════╣"
echo -e "║  🔌 API FastAPI   → http://localhost:8000                    ║"
echo -e "║  📖 Swagger Docs  → http://localhost:8000/docs               ║"
echo -e "║  📊 Dashboard     → http://localhost:3000/dashboard_realtime.html ║"
echo -e "║                                                              ║"
echo -e "║  📄 Logs en direct :                                         ║"
echo -e "║     tail -f logs_api.txt                                     ║"
echo -e "║     tail -f logs_moteur.txt                                  ║"
echo -e "║                                                              ║"
echo -e "║  ⛔  Ctrl+C pour tout arrêter                               ║"
echo -e "╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# Ouvrir le navigateur automatiquement si possible
sleep 2
if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:3000/dashboard_realtime.html" 2>/dev/null &
elif command -v open &>/dev/null; then
    open "http://localhost:3000/dashboard_realtime.html" 2>/dev/null &
fi

# ── Garder le script actif + surveiller les processus ──────────────────
echo -e "${CYAN}Surveillance active — Ctrl+C pour arrêter tout${RESET}"
echo ""
while true; do
    # Vérifier que les processus tournent encore
    if [ $PID_API -ne 0 ] && ! kill -0 $PID_API 2>/dev/null; then
        echo -e "${RED}⚠ API FastAPI s'est arrêtée ! Vérifier logs_api.txt${RESET}"
        PID_API=0
    fi
    if [ $PID_MOTEUR -ne 0 ] && ! kill -0 $PID_MOTEUR 2>/dev/null; then
        echo -e "${YELLOW}⚠ Moteur de données terminé (normal si replay fini)${RESET}"
        PID_MOTEUR=0
    fi
    sleep 5
done
