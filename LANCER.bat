@echo off
setlocal enabledelayedexpansion
title MAINTENANCE PREDICTIVE - PFE ISG Bizerte

cls
echo.
echo ============================================================
echo   MAINTENANCE PREDICTIVE - PFE ISG Bizerte
echo   Surveillance de 20 capteurs IFM - Novation City
echo ============================================================
echo.

cd /d %~dp0
if not exist "logs\" mkdir "logs"

REM ── Lecture .env (host, user, database, API_KEYS) ────────────────────────────
set DB_HOST=localhost
set DB_USER=app_user
set DB_NAME=ai_cp
set DB_PASS=
set API_KEYS=
set REPLAY_N=5000

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if /i "%%A"=="MARIADB_HOST"     set DB_HOST=%%B
        if /i "%%A"=="MARIADB_USER"     set DB_USER=%%B
        if /i "%%A"=="MARIADB_DATABASE" set DB_NAME=%%B
        if /i "%%A"=="MARIADB_PASSWORD" set DB_PASS=%%B
        if /i "%%A"=="API_KEYS"         set API_KEYS=%%B
    )
)

echo   Config chargee depuis .env :
echo     Host     : !DB_HOST!
echo     User     : !DB_USER!
echo     Database : !DB_NAME!
if "!API_KEYS!"=="" (
    echo     API_KEYS : MANQUANT - les endpoints proteges retourneront 503
) else (
    echo     API_KEYS : defini (OK)
)
echo.

REM ── Mot de passe si absent du .env ───────────────────────────────────────────
if "!DB_PASS!"=="" (
    set /p DB_PASS="Mot de passe MariaDB (absent du .env) : "
)

REM ── Menu mode ────────────────────────────────────────────────────────────────
:menu
echo Choisissez le mode de lancement :
echo.
echo   1 - REPLAY SQL   (donnees historiques depuis MariaDB)
echo   2 - TEMPS REEL   (capteurs IFM en direct)
echo   3 - QUITTER
echo.
set CHOIX=
set /p CHOIX="Votre choix (1/2/3) : "

if "!CHOIX!"=="1" goto mode_replay
if "!CHOIX!"=="2" goto mode_realtime
if "!CHOIX!"=="3" exit /b 0
echo   Choix invalide, recommencez.
echo.
goto menu

:mode_replay
set REPLAY_N=5000
set /p REPLAY_N="Nb mesures a rejouer [5000] : "
if "!REPLAY_N!"=="" set REPLAY_N=5000
set MODE_LABEL=REPLAY SQL (!REPLAY_N! mesures)
set EXTRA_ARGS=--replay !REPLAY_N!
goto lancer

:mode_realtime
set MODE_LABEL=TEMPS REEL
set EXTRA_ARGS=
goto lancer

REM ── Verification environnement ───────────────────────────────────────────────
:lancer
cls
echo.
echo ============================================================
echo   MODE : !MODE_LABEL!
echo   Host : !DB_HOST! / User : !DB_USER! / DB : !DB_NAME!
echo ============================================================
echo.

echo [1/5] Base de donnees MariaDB...
set "MYSQLD_EXE=C:\Program Files\MariaDB 12.3\bin\mysqld.exe"
set "MYSQLD_INI=C:\Program Files\MariaDB 12.3\data\my.ini"
set "MARIADB_UP=0"
powershell -NoProfile -Command "exit [int](-not (Test-NetConnection -ComputerName localhost -Port 3306 -WarningAction SilentlyContinue).TcpTestSucceeded)"
if not errorlevel 1 set "MARIADB_UP=1"
if "!MARIADB_UP!"=="1" (
    echo        OK : MariaDB deja demarre sur le port 3306
) else if exist "!MYSQLD_EXE!" (
    echo        Demarrage automatique de MariaDB...
    start "MariaDB" /min "!MYSQLD_EXE!" --defaults-file="!MYSQLD_INI!"
    echo        Attente MariaDB ^(jusqu'a 15s^)...
    set "MARIADB_UP=0"
    for /l %%i in (1,1,15) do (
        if "!MARIADB_UP!"=="0" (
            ping -n 2 127.0.0.1 >nul
            powershell -NoProfile -Command "exit [int](-not (Test-NetConnection -ComputerName localhost -Port 3306 -WarningAction SilentlyContinue).TcpTestSucceeded)"
            if not errorlevel 1 set "MARIADB_UP=1"
        )
    )
    if "!MARIADB_UP!"=="1" (
        echo        OK : MariaDB demarre
    ) else (
        echo        ATTENTION : MariaDB ne repond pas apres 15s - verifie logs\logs_moteur.txt
    )
) else (
    echo        ATTENTION : MariaDB introuvable a "!MYSQLD_EXE!"
    echo        Le moteur ne pourra pas se connecter. Modifie MYSQLD_EXE dans ce script
    echo        si MariaDB est installe ailleurs.
)

echo.
echo [2/5] Python...
python --version >nul 2>&1 || (echo ERREUR : Python introuvable & pause & exit /b 1)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo        OK : %%v

echo.
echo [3/5] Virtualenv...
if not exist "venv\" (
    echo        Creation venv...
    python -m venv venv || (echo ERREUR venv & pause & exit /b 1)
)
call venv\Scripts\activate.bat >nul 2>&1
echo        OK

echo.
echo [4/5] Dependances...
pip install -r requirements.txt -q --no-warn-script-location
echo        OK

echo.
echo [5/5] Modeles ML...
if not exist "models\model_if_v3.pkl" (
    echo ERREUR : models\model_if_v3.pkl manquant
    pause & exit /b 1
)
echo        OK : IF + LOF + OCSVM + ECOD + HBOS + COPOD

REM ── Reinitialisation ─────────────────────────────────────────────────────────
echo.
echo --- Reinitialisation ---
echo [] > realtime_results.json
echo {} > anomaly_history_persist.json
type nul > logs\logs_api.txt
type nul > logs\logs_moteur.txt
type nul > logs\logs_http.txt
echo        OK

REM ── Lancement ────────────────────────────────────────────────────────────────
echo.
echo --- Lancement ---

echo [1/3] API FastAPI  -^> http://localhost:8000
REM Appel direct de venv\Scripts\python.exe (pas besoin d'activate.bat pour
REM lancer un seul script -- python resout tout seul son propre venv/site-packages
REM depuis l'emplacement de l'exe). Commande ecrite dans un .bat genere puis
REM lance directement par "start" : evite tout guillemet imbrique dans un
REM "cmd /c "...."", source d'erreurs "nom de fichier ... incorrect" fragiles
REM et dependantes du contexte (piege deja rencontre avec l'ancienne version
REM qui passait par cmd /c "call activate.bat && ...").
(
    echo @echo off
    echo cd /d "%%~dp0"
    echo venv\Scripts\python.exe api_unified_pythagore.py ^> logs\logs_api.txt 2^>^&1
) > "%~dp0_run_api.bat"
start "API FastAPI" /min "%~dp0_run_api.bat"

echo        Attente 15 secondes...
timeout /t 15 /nobreak >nul
python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health',timeout=5)" >nul 2>&1 && echo        OK : API operationnelle || echo        ATTENTION : API lente (voir logs\logs_api.txt)

echo.
echo [2/3] Moteur       -^> !MODE_LABEL!
(
    echo @echo off
    echo cd /d "%%~dp0"
    echo venv\Scripts\python.exe realtime_mariadb.py --host !DB_HOST! --user !DB_USER! --password !DB_PASS! --database !DB_NAME! !EXTRA_ARGS! ^> logs\logs_moteur.txt 2^>^&1
) > "%~dp0_run_moteur.bat"
start "Moteur donnees" /min "%~dp0_run_moteur.bat"
echo        Logs : logs\logs_moteur.txt

echo.
echo [3/3] Dashboard    -^> http://localhost:3000
(
    echo @echo off
    echo cd /d "%%~dp0"
    echo venv\Scripts\python.exe -m http.server 3000 ^> logs\logs_http.txt 2^>^&1
) > "%~dp0_run_dashboard.bat"
start "Dashboard" /min "%~dp0_run_dashboard.bat"

REM ── Recap ────────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   SYSTEME DEMARRE - !MODE_LABEL!
echo ============================================================
echo.
echo   Connexion             : http://localhost:3000/login.html
echo   Dashboard (principal) : http://localhost:3000/dashboard_realtime.html
echo   Dashboard flotte      : http://localhost:3000/dashboard_predictive.html
echo   API Swagger/Docs      : http://localhost:8000/docs
echo   API Health            : http://localhost:8000/health
echo.
echo   Reseau local : http://192.168.100.62:3000/login.html
echo.
echo   Arreter tout : taskkill /F /IM python.exe
echo ============================================================
echo.

timeout /t 5 /nobreak >nul
start "" "http://localhost:3000/login.html"

REM ── Surveillance logs ────────────────────────────────────────────────────────
:watch_loop
echo --- %time% --- !MODE_LABEL! ---
echo [API]
powershell -command "if(Test-Path 'logs\logs_api.txt'){Get-Content 'logs\logs_api.txt' -Tail 2 -EA SilentlyContinue|%%{'   '+$_}}"
echo [MOTEUR]
powershell -command "if(Test-Path 'logs\logs_moteur.txt'){Get-Content 'logs\logs_moteur.txt' -Tail 3 -EA SilentlyContinue|%%{'   '+$_}}"
echo.
timeout /t 5 /nobreak >nul
goto watch_loop
