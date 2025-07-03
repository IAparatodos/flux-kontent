# backend.py (FastAPI)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, base64, io
import replicate # Asegúrate de que esta línea esté presente

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tudominio.com"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    imageBase64: str

@app.post("/api/editar-imagen")
async def editar(payload: Payload):
    prompt = os.getenv("FLUX_PROMPT")

    b64 = payload.imageBase64.strip()
    b64 += "=" * (-len(b64) % 4)
    img_bytes = base64.b64decode(b64)

    input_image_b64_prefixed = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

    try:
        # ¡IMPORTANTE! Hemos actualizado el ID/Versión del modelo aquí
        output_url = replicate.run(
            "black-forest-labs/flux-kontext-pro:ks1w6tyk9nrma0cq6ycacv92xm", # Modelo y versión confirmados
            input={
                "image": input_image_b64_prefixed,
                "prompt": prompt,
                "negative_prompt": "cartoon, painting, illustration, low quality, bad quality, ugly, blurry, deformed",
                "guidance_scale": 2.5
                # Asegúrate de revisar la página de 'flux-kontext-pro' en Replicate para confirmar
                # todos los parámetros de entrada y si 'guidance_scale' es válido para esta versión Pro.
            }
        )

        import requests # Asegúrate de que esta línea esté presente si no la tienes
        response = requests.get(output_url)
        response.raise_for_status()
        modified_image_bytes = response.content

        mod_b64 = base64.b64encode(modified_image_bytes).decode("utf-8")
        return {"modifiedImage": mod_b64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen con Replicate: {str(e)}")
