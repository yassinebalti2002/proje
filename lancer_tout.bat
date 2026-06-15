@echo off
REM ╔══════════════════════════════════════════════════════════════════════╗
REM ║   LANCEMENT COMPLET EN UNE SEULE COMMANDE — WINDOWS                 ║
REM ║   Lance en parallèle :                                              ║
REM ║     [1] API FastAPI          → http://localhost:8000                ║
REM ║     [2] Moteur realtime      → MariaDB IoT                          ║
REM ║     [3] Serveur Dashboard    → http://localhost:3000                ║
REM ║                                                                      ║
REM ║   Usage :                                                            ║
REM ║     lancer_tout.bat              (temps réel)                       ║
REM ║     lancer_tout.bat replay 100   (replay 100 mesures)               ║
REM ║     lancer_tout.bat simulateur   (sans capteurs ni MariaDB)         ║
REM ║                                                                      ║
REM ║   Arrêt : fermer cette fenêtre — arrête TOUT                       ║
REM ╚══════════════════════════════════════════════════════════════════════╝

setlocal enabledelayedexpansion
title MAINTENANCE PREDICTIVE — SYSTEME COMPLET

cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   MAINTENANCE PREDICTIVE — LANCEMENT COMPLET                ║
echo ║   %date% %time%
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM ── Paramètres ────────────────────────────────────────────────────────
set MODE=%1
set REPLAY_N=%2
if "%MODE%"=="" set MODE=realtime

REM ── Lecture config.py (IP IoT, ports) ────────────────────────────────
for /f "tokens=*" %%i in ('python -c "from config import MARIADB_HOST; print(MARIADB_HOST)" 2^>nul') do set IOT_IP=%%i
for /f "tokens=*" %%i in ('python -c "from config import API_PORT; print(API_PORT)" 2^>nul') do set API_PORT=%%i
for /f "tokens=*" %%i in ('python -c "from config import DASHBOARD_PORT; print(DASHBOARD_PORT)" 2^>nul') do set DASH_PORT=%%i
if "!IOT_IP!"==""  set IOT_IP=192.168.1.50
if "!API_PORT!"=="" set API_PORT=8000
if "!DASH_PORT!"=="" set DASH_PORT=3000

REM ── Étape 1 : Python ───────────────────────────────────────────────────
echo [1/5] Vérification Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo   ERREUR : Python non trouvé — installez Python 3.10+
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   OK : %%v

REM ── Étape 2 : Virtualenv ───────────────────────────────────────────────
echo.
echo [2/5] Virtualenv...
if not exist "venv\" (
    echo   Creation du venv...
    python -m venv venv
)
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo   ERREUR activation venv
    pause
    exit /b 1
)
echo   OK : Virtualenv active

REM ── Étape 3 : Dépendances ──────────────────────────────────────────────
echo.
echo [3/5] Dependances...
pip install -q -r requirements.txt >nul 2>&1
echo   OK : Dependances installees

REM ── Étape 4 : Vérifier fichiers ────────────────────────────────────────
echo.
echo [4/5] Verification fichiers...
if not exist "api_unified_pythagore.py" (
    echo   ERREUR : api_unified_pythagore.py manquant
    pause & exit /b 1
)
if not exist "realtime_mariadb.py" (
    echo   ERREUR : realtime_mariadb.py manquant
    pause & exit /b 1
)
if not exist "models\model_if_v3.pkl" (
    echo   ERREUR : dossier models/ manquant ou incomplet
    pause & exit /b 1
)
echo   OK : Tous les fichiers presents

REM ── Étape 5 : Lancement en parallèle ─────────────────────────────────
echo.
echo [5/5] Lancement des 3 processus...
echo.

REM --- Processus 1 : API FastAPI (fenêtre minimisée en arrière-plan) ---
echo   [1/3] API FastAPI sur http://localhost:!API_PORT!
start "API FastAPI" /min cmd /c "call venv\Scripts\activate.bat && python api_unified_pythagore.py > logs_api.txt 2>&1"
echo         Logs : logs_api.txt

REM Attendre 8s que l'API démarre avant de lancer le moteur
echo         Attente demarrage API (8s)...
timeout /t 8 /nobreak >nul

REM --- Processus 2 : Moteur de données (fenêtre minimisée) ---
if "%MODE%"=="simulateur" (
    echo   [2/3] Simulateur ^(mode sans capteurs^)
    start "Moteur Simulateur" /min cmd /c "call venv\Scripts\activate.bat && python realtime_simulator.py --scenario aleatoire > logs_moteur.txt 2>&1"
) else if "%MODE%"=="replay" (
    echo   [2/3] Moteur MariaDB REPLAY %REPLAY_N% mesures
    start "Moteur Replay" /min cmd /c "call venv\Scripts\activate.bat && python realtime_mariadb.py --replay %REPLAY_N% > logs_moteur.txt 2>&1"
) else (
    echo   [2/3] Moteur MariaDB TEMPS REEL ^(!IOT_IP!^)
    start "Moteur Realtime" /min cmd /c "call venv\Scripts\activate.bat && python realtime_mariadb.py > logs_moteur.txt 2>&1"
)
echo         Logs : logs_moteur.txt

REM --- Processus 3 : Serveur HTTP Dashboard (fenêtre minimisée) ---
echo   [3/3] Dashboard HTTP sur http://localhost:!DASH_PORT!
start "Dashboard HTTP" /min cmd /c "call venv\Scripts\activate.bat && python -m http.server !DASH_PORT! > logs_http.txt 2>&1"
echo         Logs : logs_http.txt

REM ── Résumé final ───────────────────────────────────────────────────────
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   OK SYSTEME DEMARRE COMPLETEMENT                           ║
echo ╠══════════════════════════════════════════════════════════════╣
echo ║  API FastAPI   --^> http://localhost:!API_PORT!                    ║
echo ║  Swagger Docs  --^> http://localhost:!API_PORT!/docs               ║
echo ║  Dashboard     --^> http://localhost:!DASH_PORT!/dashboard_realtime.html
echo ║                                                              ║
echo ║  Logs en direct (dans un autre terminal) :                  ║
echo ║    type logs_api.txt                                        ║
echo ║    type logs_moteur.txt                                     ║
echo ║                                                              ║
echo ║  Pour tout arreter : fermez cette fenetre                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Ouvrir le dashboard dans le navigateur automatiquement
timeout /t 3 /nobreak >nul
start "" "http://localhost:!DASH_PORT!/dashboard_realtime.html"

REM ── Surveillance : afficher les logs en temps réel ────────────────────
echo Surveillance active. Appuyez sur Ctrl+C pour tout arreter.
echo.

:watch_loop
    REM Afficher les 3 dernières lignes de chaque log toutes les 5s
    echo ──────────────────────────────────────────────── %time%
    echo [API]
    for /f "skip=0 delims=" %%a in ('type logs_api.txt 2^>nul') do set "last_api=%%a"
    echo   !last_api!
    echo [MOTEUR]
    for /f "skip=0 delims=" %%a in ('type logs_moteur.txt 2^>nul') do set "last_mot=%%a"
    echo   !last_mot!
    timeout /t 5 /nobreak >nul
goto watch_loop
