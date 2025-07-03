# backend.py (FastAPI)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os, base64, io

# Carga las variables de entorno desde .env
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tudominio.com"],  # ajusta al dominio de tu WP
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Inicializa el cliente de HF con tu token
hf = InferenceClient(token=os.getenv("HF_TOKEN"))

class Payload(BaseModel):
    imageBase64: str

@app.post("/api/editar-imagen")
async def editar(payload: Payload):
    prompt = os.getenv("FLUX_PROMPT")
    # Limpia padding Base64
    b64 = payload.imageBase64.strip()
    b64 += "=" * (-len(b64) % 4)
    img_bytes = base64.b64decode(b64)

    try:
        # Llama al modelo correcto
        output_img = hf.image_to_image(
            img_bytes,
            prompt=prompt,
            model="black-forest-labs/FLUX.1-Kontext-dev",
            guidance_scale=2.5
        )  # devuelve un PIL.Image

        # Serializa la imagen a Base64 de nuevo
        buf = io.BytesIO()
        output_img.save(buf, format="PNG")
        mod_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"modifiedImage": mod_b64}

    except Exception as e:
        # Devuelve el error real en la respuesta
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
