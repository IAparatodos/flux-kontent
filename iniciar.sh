#!/bin/bash

echo "🎬 Iniciando Agenda de Videos..."
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no está instalado. Por favor instala Python 3.8 o superior."
    exit 1
fi

# Instalar dependencias si es necesario
echo "📦 Instalando dependencias..."
pip3 install -q -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt

echo ""
echo "✅ Todo listo! La agenda está corriendo en:"
echo ""
echo "   🖥️  Ordenador: http://localhost:8000"
echo "   📱 Móvil: http://$(hostname -I | awk '{print $1}'):8000"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo ""

# Iniciar servidor
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
