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
    allow_origins=["https://www.codigoadria.com/"],  # ¡IMPORTANTE! Ajusta esto al dominio exacto de tu WordPress
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    imageBase66: str

@app.post("/api/editar-imagen")
async def editar(payload: Payload):
    prompt = os.getenv("FLUX_PROMPT")

    # EXTRAEMOS LA CADENA BASE64 PURA eliminando el prefijo
    # Por ejemplo, de "data:image/png;base64,iVBORw0..." tomamos solo "iVBORw0..."
    # Usamos .split(",", 1)[1] para dividir solo por la primera coma y tomar la segunda parte.
    b66 = payload.imageBase66.split(",", 1)[1].strip()

    # Asegura padding correcto del Base64 (esto es correcto y debe mantenerse)
    b66 += "=" * (-len(b66) % 4)
    
    try:
        img_bytes = base64.b64decode(b66) # Ahora 'b66' solo contiene la cadena Base64 pura

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
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen con Replicate: {str(e)}")
