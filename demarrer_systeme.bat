@echo off
REM ╔════════════════════════════════════════════════════════════════════════╗
REM ║  DEMARRAGE AUTOMATIQUE — SYSTÈME TEMPS RÉEL                           ║
REM ║  Windows PowerShell / CMD                                             ║
REM ║  Usage : demarrer_systeme.bat                                         ║
REM ╚════════════════════════════════════════════════════════════════════════╝

setlocal enabledelayedexpansion

cls
echo.
echo ╔════════════════════════════════════════════════════════════════════════╗
echo ║  MOTEUR TEMPS RÉEL — MariaDB IoT ^> API FastAPI                      ║
echo ║  Machine : %COMPUTERNAME%                                             ║
echo ║  Date : %date% %time%                                                 ║
echo ╚════════════════════════════════════════════════════════════════════════╝
echo.

REM ── Vérifier Python ──────────────────────────────────────────────────────
echo [1/4] Vérification Python...
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Python non trouvé — installez Python 3.10+
    echo.
    pause
    exit /b 1
)
echo ✅ Python détecté
echo.

REM ── Vérifier/Créer virtualenv ────────────────────────────────────────────
echo [2/4] Préparation virtualenv...
if not exist "venv\" (
    echo    Création venv...
    python -m venv venv
)
call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo ❌ Échec activation venv
    pause
    exit /b 1
)
echo ✅ Virtualenv activé
echo.

REM ── Installer dépendances ───────────────────────────────────────────────
echo [3/4] Installation dépendances...
pip install -q -r requirements.txt >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Échec installation dépendances
    echo    Lancez : pip install -r requirements.txt
    pause
    exit /b 1
)
echo ✅ Dépendances OK
echo.

REM ── Diagnostic MariaDB ──────────────────────────────────────────────────
echo [4/4] Diagnostic MariaDB...
python realtime_mariadb.py --diagnostic
if !errorlevel! neq 0 (
    echo.
    echo ⚠️  Diagnostic échoué — vérifier configuration MariaDB
    pause
    exit /b 1
)
echo.

REM ── Choix mode de lancement ─────────────────────────────────────────────
echo.
echo ╔════════════════════════════════════════════════════════════════════════╗
echo ║  MODES DE LANCEMENT                                                    ║
echo ╚════════════════════════════════════════════════════════════════════════╝
echo.
:choice_menu
echo 1 = Temps réel (attend données capteurs)
echo 2 = Replay (rejeu 50 dernières mesures stockées)
echo 3 = Replay personnalisé (N mesures)
echo 4 = Configuration avancée
echo 5 = Quitter
echo.
set /p CHOICE="Choix (1-5) : "

if "%CHOICE%"=="1" goto realtime
if "%CHOICE%"=="2" goto replay50
if "%CHOICE%"=="3" goto replay_custom
if "%CHOICE%"=="4" goto advanced
if "%CHOICE%"=="5" goto end
goto invalid

:realtime
echo.
echo ▶️  Démarrage MODE TEMPS RÉEL...
echo    Données capteurs IFM en direct depuis 192.168.1.50:3306
echo.
python realtime_mariadb.py
goto end

:replay50
echo.
echo ▶️  Démarrage MODE REPLAY (50 mesures)...
echo.
python realtime_mariadb.py --replay 50
goto end

:replay_custom
set /p N="Nombre de mesures à rejouer : "
python realtime_mariadb.py --replay %N%
goto end

:advanced
echo.
echo ╔════════════════════════════════════════════════════════════════════════╗
echo ║  CONFIGURATION AVANCÉE                                                ║
echo ╚════════════════════════════════════════════════════════════════════════╝
echo.
set /p HOST="IP MariaDB [192.168.1.50] : "
if "%HOST%"=="" set HOST=192.168.1.50

set /p USER="Utilisateur [root] : "
if "%USER%"=="" set USER=root

set /p DB="Base de données [ai_cp] : "
if "%DB%"=="" set DB=ai_cp

set /p WINDOW="Taille fenêtre [10] : "
if "%WINDOW%"=="" set WINDOW=10

set /p POLL="Intervalle poll [2.0] : "
if "%POLL%"=="" set POLL=2.0

echo.
echo ▶️  Démarrage avec config...
echo    Host: %HOST% | User: %USER% | DB: %DB%
echo    Window: %WINDOW% | Poll: %POLL%s
echo.
python realtime_mariadb.py --host %HOST% --user %USER% --database %DB% --window %WINDOW% --poll %POLL%
goto end

:invalid
echo ❌ Choix invalide
pause
goto choice_menu

:end
echo.
echo Arrêt — À bientôt! 👋
pause
