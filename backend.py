# backend.py (FastAPI)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime, date
from typing import Optional, List
import os, base64, io
import replicate
import requests

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import enum

load_dotenv()

# Database setup
DATABASE_URL = "sqlite:///./videos_agenda.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enum for video status
class VideoStatus(str, enum.Enum):
    PENDIENTE = "pendiente"
    EN_PROGRESO = "en_progreso"
    COMPLETADO = "completado"

# Video model for database
class VideoModel(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    titulo = Column(String, nullable=False)
    descripcion = Column(String)
    status = Column(SQLEnum(VideoStatus), default=VideoStatus.PENDIENTE)
    prioridad = Column(Integer, default=1)  # 1=baja, 2=media, 3=alta
    fecha_limite = Column(Date, nullable=True)
    fecha_creacion = Column(DateTime, default=datetime.utcnow)
    url_video = Column(String, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass  # El directorio static se creará después

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.codigoadria.com", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Pydantic models for API
class VideoCreate(BaseModel):
    titulo: str
    descripcion: Optional[str] = None
    status: VideoStatus = VideoStatus.PENDIENTE
    prioridad: int = 1
    fecha_limite: Optional[date] = None
    url_video: Optional[str] = None

class VideoUpdate(BaseModel):
    titulo: Optional[str] = None
    descripcion: Optional[str] = None
    status: Optional[VideoStatus] = None
    prioridad: Optional[int] = None
    fecha_limite: Optional[date] = None
    url_video: Optional[str] = None

class VideoResponse(BaseModel):
    id: int
    titulo: str
    descripcion: Optional[str]
    status: VideoStatus
    prioridad: int
    fecha_limite: Optional[date]
    fecha_creacion: datetime
    url_video: Optional[str]

    class Config:
        from_attributes = True

class Payload(BaseModel):
    imageBase66: str

# === ENDPOINTS DE AGENDA DE VIDEOS ===

@app.get("/")
async def root():
    """Servir la página principal de la agenda"""
    return FileResponse("static/index.html")

@app.get("/api/videos", response_model=List[VideoResponse])
async def obtener_videos(db: Session = Depends(get_db)):
    """Obtener todos los videos"""
    videos = db.query(VideoModel).order_by(VideoModel.prioridad.desc(), VideoModel.fecha_creacion.desc()).all()
    return videos

@app.get("/api/videos/{video_id}", response_model=VideoResponse)
async def obtener_video(video_id: int, db: Session = Depends(get_db)):
    """Obtener un video específico por ID"""
    video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video no encontrado")
    return video

@app.post("/api/videos", response_model=VideoResponse, status_code=201)
async def crear_video(video: VideoCreate, db: Session = Depends(get_db)):
    """Crear un nuevo video en la agenda"""
    db_video = VideoModel(**video.dict())
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video

@app.put("/api/videos/{video_id}", response_model=VideoResponse)
async def actualizar_video(video_id: int, video: VideoUpdate, db: Session = Depends(get_db)):
    """Actualizar un video existente"""
    db_video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
    if not db_video:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    # Actualizar solo los campos proporcionados
    for field, value in video.dict(exclude_unset=True).items():
        setattr(db_video, field, value)

    db.commit()
    db.refresh(db_video)
    return db_video

@app.delete("/api/videos/{video_id}")
async def eliminar_video(video_id: int, db: Session = Depends(get_db)):
    """Eliminar un video"""
    db_video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
    if not db_video:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    db.delete(db_video)
    db.commit()
    return {"message": "Video eliminado exitosamente"}

# === ENDPOINT DE EDICIÓN DE IMÁGENES (ORIGINAL) ===

# Límite de tamaño de archivo en bytes (1MB)
MAX_FILE_SIZE_BYTES_SERVER = 1 * 1024 * 1024 # 1 MB

@app.post("/api/editar-imagen")
async def editar(payload: Payload):
    prompt = os.getenv("FLUX_PROMPT")

    # EXTRAEMOS LA CADENA BASE64 PURA eliminando el prefijo
    # Por ejemplo, de "data:image/png;base64,iVBORw0..." tomamos solo "iVBORw0..."
    b66_pure = payload.imageBase66.split(",", 1)[1].strip()

    # Asegura padding correcto del Base64
    b66_pure += "=" * (-len(b66_pure) % 4)
    
    try:
        img_bytes = base64.b64decode(b66_pure) # Decodifica a bytes

        # ¡NUEVA COMPROBACIÓN! Limitar el tamaño del archivo en el servidor
        if len(img_bytes) > MAX_FILE_SIZE_BYTES_SERVER:
            raise HTTPException(
                status_code=413, # 413 Payload Too Large
                detail=f"El archivo es demasiado grande. El tamaño máximo permitido es 1MB. Tu archivo mide {(len(img_bytes) / (1024 * 1024)):.2f}MB."
            )

        # Replicate necesita la imagen de entrada en Base64 con el prefijo de tipo de dato
        # Lo reconstruimos aquí con los bytes decodificados para asegurarnos de que sea correcto.
        # Es importante que el formato (png/jpeg) sea el correcto. Asumimos png.
        input_image_b66_prefixed = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        output_url = replicate.run(
            "black-forest-labs/flux-kontext-pro",
            input={
                "input_image": input_image_b66_prefixed,
                "prompt": prompt,
                "negative_prompt": "cartoon, painting, illustration, low quality, bad quality, ugly, blurry, deformed",
                "aspect_ratio": "match_input_image",
                "prompt_upsampling": False,
                "output_format": "png"  
            }
        )
        
        response = requests.get(output_url)
        response.raise_for_status()
        modified_image_bytes = response.content

        mod_b66 = base64.b64encode(modified_image_bytes).decode("utf-8")
        return {"modifiedImage": mod_b66}

    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        # Si ya es un HTTPException (como el 413), lo relanza. Si no, crea un 500.
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen con Replicate: {str(e)}")

