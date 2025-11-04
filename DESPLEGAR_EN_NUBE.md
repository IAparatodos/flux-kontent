# 🚀 Desplegar Agenda de Videos en la Nube (GRATIS)

## ⚡ Opción 1: Render.com (Recomendado - Más Fácil)

### Paso 1: Crear cuenta en Render
1. Ve a https://render.com
2. Haz clic en "Get Started" o "Sign Up"
3. Regístrate con tu cuenta de GitHub (o email)

### Paso 2: Conectar tu repositorio
1. En el dashboard de Render, haz clic en "New +"
2. Selecciona "Web Service"
3. Conecta tu cuenta de GitHub si aún no lo has hecho
4. Busca el repositorio: `flux-kontent`
5. Haz clic en "Connect"

### Paso 3: Configurar el servicio
Completa los campos así:

- **Name:** `agenda-videos-adri` (o el nombre que prefieras)
- **Region:** Selecciona la más cercana (Frankfurt para Europa, Oregon para América)
- **Branch:** `claude/interactive-video-agenda-011CUjrR2dio7vGPw2GFUmtg`
- **Root Directory:** (déjalo vacío)
- **Runtime:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn backend:app --host 0.0.0.0 --port $PORT`
- **Instance Type:** Free (gratis)

### Paso 4: Variables de entorno (Opcional)
Si quieres usar la funcionalidad de edición de imágenes:
1. Haz clic en "Advanced"
2. Agrega una variable:
   - **Key:** `FLUX_PROMPT`
   - **Value:** Tu prompt personalizado

### Paso 5: ¡Desplegar!
1. Haz clic en "Create Web Service"
2. Espera 3-5 minutos mientras Render despliega tu aplicación
3. Una vez completado, verás una URL como: `https://agenda-videos-adri.onrender.com`

### 🎉 ¡Listo! Accede desde cualquier lugar

Abre la URL en cualquier dispositivo (móvil, tablet, PC) y tendrás acceso a tu agenda.

---

## ⚡ Opción 2: Railway.app (También fácil y rápido)

### Paso 1: Crear cuenta
1. Ve a https://railway.app
2. Haz clic en "Start a New Project"
3. Inicia sesión con GitHub

### Paso 2: Desplegar desde GitHub
1. Haz clic en "Deploy from GitHub repo"
2. Selecciona `flux-kontent`
3. Selecciona el branch: `claude/interactive-video-agenda-011CUjrR2dio7vGPw2GFUmtg`

### Paso 3: Configurar
1. Railway detectará automáticamente que es una app Python
2. Haz clic en tu proyecto → Settings → Generate Domain
3. Copia la URL generada

### 🎉 ¡Ya está en línea!

---

## ⚡ Opción 3: Fly.io (Para usuarios avanzados)

```bash
# Instalar flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Desplegar
fly launch
fly deploy
```

---

## 📝 Notas Importantes

### 🔄 Actualizaciones automáticas
Cuando hagas cambios en GitHub y hagas push, Render/Railway actualizarán automáticamente tu aplicación.

### 💾 Base de datos
La base de datos SQLite se perderá cada vez que el servicio se reinicie en el plan gratuito. Si quieres datos persistentes, puedes:
- Actualizar a un plan con disco persistente (~$1/mes)
- Usar una base de datos PostgreSQL gratuita de Render

### 🌐 Dominio personalizado
Puedes conectar tu propio dominio (ej: agenda.codigoadria.com) en la configuración del servicio.

### 💤 Suspensión en plan gratuito
Los planes gratuitos se suspenden después de 15 minutos de inactividad, pero se reactivan automáticamente cuando alguien accede (toma 30-60 segundos).

---

## 🆘 ¿Problemas?

### Error: "Build failed"
- Verifica que `requirements.txt` esté en la raíz del proyecto
- Asegúrate de que el branch seleccionado sea el correcto

### Error: "Application failed to start"
- Verifica el Start Command: `uvicorn backend:app --host 0.0.0.0 --port $PORT`
- Revisa los logs en el dashboard de Render/Railway

### No puedo acceder a la URL
- Espera 3-5 minutos después del despliegue
- Verifica que el servicio esté "Running" en el dashboard

---

## 🎯 Recomendación

**Usa Render.com** - Es el más fácil, con mejor interfaz y documentación en español.

---

¿Necesitas ayuda? Solo pregúntame y te guío paso a paso.
