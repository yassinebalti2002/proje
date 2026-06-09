@echo off
setlocal enabledelayedexpansion
title MAINTENANCE PREDICTIVE - PFE ISG Bizerte
chcp 65001 >nul 2>&1

cls
echo.
echo ============================================================
echo   MAINTENANCE PREDICTIVE - PFE ISG Bizerte
echo   Surveillance de 20 capteurs IFM - Novation City
echo   Donnees reelles - Base MySQL ai_cp
echo ============================================================
echo.

cd /d "%~dp0"

REM ============================================================
REM  MENU PRINCIPAL
REM ============================================================
:menu
echo Choisissez votre mode :
echo.
echo   1 - REPLAY SQL   (donnees reelles historiques - MySQL local)
echo   2 - TEMPS REEL   (vrais capteurs IFM sur le reseau local)
echo   3 - QUITTER
echo.
set /p CHOIX="Votre choix (1/2/3) : "

if "%CHOIX%"=="1" goto mode_replay
if "%CHOIX%"=="2" goto mode_realtime
if "%CHOIX%"=="3" exit /b 0
echo Choix invalide. Recommencez.
echo.
goto menu

REM ============================================================
REM  MODE 1 : REPLAY SQL (donnees reelles historiques)
REM ============================================================
:mode_replay
cls
echo.
echo --- MODE REPLAY SQL (DONNEES REELLES) -------------------------
echo Rejoue les dernieres mesures reelles depuis MySQL ai_cp
echo Capteurs : 20 capteurs IFM (6e0c1740, b2acdf45, aa7b02a1...)
echo Periode  : 10/02/2026 -> 21/03/2026 (1 648 886 mesures)
echo.

REM Valeurs pre-configurees MySQL local
set DB_HOST=localhost
set DB_USER=root
set DB_PASS=yassine2019
set DB_NAME=ai_cp
set REPLAY_N=5000

set /p REPLAY_N="Nb mesures a rejouer [5000] : "
if "!REPLAY_N!"=="" set REPLAY_N=5000

set MODE_LABEL=REPLAY SQL (!REPLAY_N! mesures reelles)
set MOTEUR_CMD=python realtime_mariadb.py --host !DB_HOST! --user !DB_USER! --password !DB_PASS! --database !DB_NAME! --replay !REPLAY_N! --window 5
goto check_env

REM ============================================================
REM  MODE 2 : TEMPS REEL (vrais capteurs IFM)
REM ============================================================
:mode_realtime
cls
echo.
echo --- MODE TEMPS REEL (CAPTEURS IFM) ----------------------------
echo Connexion directe au serveur IoT avec capteurs physiques
echo.
set DB_HOST=localhost
set /p DB_HOST="IP serveur IoT [localhost] : "
if "!DB_HOST!"=="" set DB_HOST=localhost

set DB_USER=root
set /p DB_USER="Utilisateur    [root]      : "
if "!DB_USER!"=="" set DB_USER=root

set DB_PASS=yassine2019
set /p DB_PASS="Mot de passe   [yassine2019] : "
if "!DB_PASS!"=="" set DB_PASS=yassine2019

set DB_NAME=ai_cp
set /p DB_NAME="Base de donnees [ai_cp]    : "
if "!DB_NAME!"=="" set DB_NAME=ai_cp

set MODE_LABEL=TEMPS REEL (capteurs IFM)
set MOTEUR_CMD=python realtime_mariadb.py --host !DB_HOST! --user !DB_USER! --password !DB_PASS! --database !DB_NAME!
goto check_env

REM ============================================================
REM  VERIFICATION ENVIRONNEMENT
REM ============================================================
:check_env
cls
echo.
echo ============================================================
echo   MODE : !MODE_LABEL!
echo ============================================================
echo.

echo [1/4] Verification Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR : Python non trouve. Installez Python 3.10+
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo        OK : %%v

echo.
echo [2/4] Preparation virtualenv...
if not exist "venv\" (
    echo        Creation venv...
    python -m venv venv
    if errorlevel 1 (
        echo ERREUR : Impossible de creer le venv
        pause & exit /b 1
    )
)
call venv\Scripts\activate.bat >nul 2>&1
if errorlevel 1 (
    echo ERREUR : Activation venv echouee. Supprimez venv\ et relancez.
    pause & exit /b 1
)
echo        OK : venv active

echo.
echo [3/4] Installation dependances...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERREUR : pip install a echoue.
    pause & exit /b 1
)
echo        OK : dependances installes

echo.
echo [4/4] Verification modeles ML...
if not exist "models\model_if_v3.pkl" (
    echo ERREUR : models\model_if_v3.pkl manquant
    echo Lancez d'abord : python train_model_v3_unsupervised.py
    pause & exit /b 1
)
echo        OK : modeles ML detectes (IF + LOF + OCSVM + ECOD)

REM ============================================================
REM  REINITIALISATION DES FICHIERS DE DONNEES
REM ============================================================
echo.
echo --- Reinitialisation des donnees ------------------------------
echo [] > realtime_results.json
echo {} > anomaly_history_persist.json
type nul > logs_api.txt
type nul > logs_moteur.txt
type nul > logs_http.txt
echo        OK : fichiers reinitialises

REM ============================================================
REM  LANCEMENT DES 3 PROCESSUS
REM ============================================================
echo.
echo --- Lancement des processus -----------------------------------
echo.

echo [1/3] API FastAPI       -> http://localhost:8000
start "API FastAPI" /min cmd /c "cd /d "%~dp0" && call venv\Scripts\activate.bat && python api_unified_pythagore.py > logs_api.txt 2>&1"

echo        Attente demarrage API (12 secondes)...
timeout /t 12 /nobreak >nul

python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" >nul 2>&1
if errorlevel 1 (
    echo        ATTENTION : API ne repond pas encore (voir logs_api.txt)
) else (
    echo        OK : API operationnelle
)

echo.
echo [2/3] Moteur donnees    -> !MODE_LABEL!
start "Moteur donnees" /min cmd /c "cd /d "%~dp0" && call venv\Scripts\activate.bat && !MOTEUR_CMD! > logs_moteur.txt 2>&1"
echo        Logs : logs_moteur.txt

echo.
echo [3/3] Dashboard HTTP    -> http://localhost:3000
start "Dashboard HTTP" /min cmd /c "cd /d "%~dp0" && python -m http.server 3000 > logs_http.txt 2>&1"

REM ============================================================
REM  RECAPITULATIF
REM ============================================================
echo.
echo ============================================================
echo   SYSTEME DEMARRE - !MODE_LABEL!
echo ============================================================
echo.
echo   Dashboard predictif  : http://localhost:3000/dashboard_predictive.html
echo   API Swagger/Docs     : http://localhost:8000/docs
echo   API Health check     : http://localhost:8000/health
echo.
echo   Reseau local (telephone/tablette) :
echo   http://192.168.100.62:3000/dashboard_predictive.html
echo.
echo   Attendre ~60s pour les premieres predictions reelles
echo.
echo   Pour tout arreter : taskkill /F /IM python.exe
echo ============================================================
echo.

timeout /t 5 /nobreak >nul
start "" "http://localhost:3000/dashboard_predictive.html"

REM ============================================================
REM  SURVEILLANCE LOGS EN DIRECT
REM ============================================================
:watch_loop
echo --- %time% --- !MODE_LABEL! ---
echo [API]
powershell -command "if(Test-Path 'logs_api.txt'){Get-Content 'logs_api.txt' -Tail 2 -ErrorAction SilentlyContinue | ForEach-Object {'   '+$_}}"
echo [MOTEUR]
powershell -command "if(Test-Path 'logs_moteur.txt'){Get-Content 'logs_moteur.txt' -Tail 3 -ErrorAction SilentlyContinue | ForEach-Object {'   '+$_}}"
echo.
timeout /t 5 /nobreak >nul
goto watch_loop
