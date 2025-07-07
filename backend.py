# backend.py (FastAPI)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, base64, io
import replicate
import requests

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.codigoadria.com"], # Asegúrate de que este sea tu dominio de WordPress
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    imageBase66: str

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

