@echo off
echo 🎬 Iniciando Agenda de Videos...
echo.

REM Instalar dependencias
echo 📦 Instalando dependencias...
pip install -q -r requirements.txt

echo.
echo ✅ Todo listo! La agenda está corriendo en:
echo.
echo    🖥️  http://localhost:8000
echo.
echo Presiona Ctrl+C para detener el servidor
echo.

REM Iniciar servidor
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
