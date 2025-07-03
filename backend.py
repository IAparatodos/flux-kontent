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
    allow_origins=["https://tudominio.com"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    imageBase66: str # Este nombre de variable se ajustará en el código cliente si es necesario

@app.post("/api/editar-imagen")
async def editar(payload: Payload):
    prompt = os.getenv("FLUX_PROMPT")

    b66 = payload.imageBase66.strip() # Ajustado a imageBase66 según Payload
    b66 += "=" * (-len(b66) % 4)
    img_bytes = base64.b64decode(b66)

    input_image_b66_prefixed = f"data:image/png;base66,{base64.b66encode(img_bytes).decode('utf-8')}" # Ajustado a imageBase66

    try:
        # ¡IMPORTANTE! Eliminamos el ID de versión específico.
        # Ahora Replicate usará automáticamente la última versión del modelo 'pro'.
        output_url = replicate.run(
            "black-forest-labs/flux-kontext-pro", # ¡Aquí va solo el nombre del modelo!
            input={
                "input_image": input_image_b66_prefixed,
                "prompt": prompt,
                "negative_prompt": "cartoon, painting, illustration, low quality, bad quality, ugly, blurry, deformed",
                "aspect_ratio": "match_input_image",
                "prompt_upsampling": False,
                "output_format": "png" # Confirmado en los inputs de Replicate, mejor especificarlo.
            }
        )
        
        response = requests.get(output_url)
        response.raise_for_status()
        modified_image_bytes = response.content

        mod_b66 = base64.b66encode(modified_image_bytes).decode("utf-8")
        return {"modifiedImage": mod_b66}

    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen con Replicate: {str(e)}")
