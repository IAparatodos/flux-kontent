# 📹 Agenda de Videos - Academia Código Adri

Aplicación web interactiva para gestionar y hacer seguimiento de los videos pendientes de la academia de código.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Características

- 📊 **Panel de estadísticas** en tiempo real
- ✅ **Gestión completa** de videos (crear, editar, eliminar)
- 🎯 **Sistema de prioridades** (baja, media, alta)
- 📅 **Fechas límite** para organizar tu trabajo
- 🔄 **Estados de progreso** (pendiente, en progreso, completado)
- 🎨 **Interfaz moderna** y responsive (móvil, tablet, PC)
- 🔗 **Enlaces a videos** de YouTube u otras plataformas
- 🎛️ **Filtros por estado** para visualización rápida

## 🚀 Inicio Rápido

### Opción 1: Desplegar en la Nube (Sin instalar nada) ⭐

La forma más fácil es desplegar en Render.com (gratis):

1. Ve a https://render.com y crea una cuenta
2. Conecta este repositorio
3. Sigue la [guía completa de despliegue](DESPLEGAR_EN_NUBE.md)
4. ¡Accede desde cualquier lugar!

### Opción 2: Ejecutar localmente (1 comando)

```bash
# Clonar el repositorio
git clone https://github.com/IAparatodos/flux-kontent.git
cd flux-kontent

# Ejecutar (Mac/Linux)
./iniciar.sh

# O en Windows
iniciar.bat
```

Luego abre: http://localhost:8000

## 📦 Instalación Manual

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar servidor
uvicorn backend:app --reload

# 3. Abrir navegador
# http://localhost:8000
```

## 🌐 Acceder desde el móvil

Si ejecutas la aplicación localmente:

1. Asegúrate de que tu móvil y PC estén en la misma WiFi
2. Ejecuta: `uvicorn backend:app --host 0.0.0.0 --port 8000`
3. Encuentra la IP de tu PC:
   - Windows: `ipconfig`
   - Mac/Linux: `ifconfig` o `hostname -I`
4. En tu móvil: `http://TU_IP:8000`

## 🛠️ Tecnologías

- **Backend:** FastAPI (Python)
- **Base de datos:** SQLite + SQLAlchemy
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **Despliegue:** Render.com / Railway / Heroku

## 📖 Documentación

- [Instrucciones detalladas](INSTRUCCIONES.md)
- [Guía de despliegue en la nube](DESPLEGAR_EN_NUBE.md)

## 🎯 Uso

### Agregar un video

1. Rellena el formulario con el título del video
2. Añade descripción, prioridad y fecha límite (opcional)
3. Haz clic en "Agregar Video"

### Gestionar videos

- **Cambiar estado:** Usa los botones de cada tarjeta
- **Filtrar:** Haz clic en los filtros superiores
- **Eliminar:** Botón rojo de eliminar

### Estadísticas

El panel superior muestra en tiempo real:
- Total de videos
- Videos pendientes
- Videos en progreso
- Videos completados

## 📸 Capturas

<img src="https://via.placeholder.com/800x500?text=Agenda+Interactiva" alt="Vista principal">

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT.

## 👤 Autor

**Academia Código Adri**

- Website: https://www.codigoadria.com

## 🆘 Soporte

¿Necesitas ayuda? Abre un [issue](https://github.com/IAparatodos/flux-kontent/issues) en GitHub.

---

⭐ Si te gusta este proyecto, dale una estrella en GitHub!
