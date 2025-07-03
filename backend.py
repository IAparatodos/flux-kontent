# backend.py (FastAPI)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, base64, io
import replicate  # Asegúrate de haber instalado 'replicate'
import requests   # Asegúrate de haber instalado 'requests'

load_dotenv()  # Carga REPLICATE_API_TOKEN y FLUX_PROMPT desde .env o variables de entorno

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tudominio.com"],  # ¡IMPORTANTE! Ajusta esto al dominio exacto de tu WordPress
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    imageBase64: str

@app.post("/api/editar-imagen")
async def editar(payload: Payload):
    prompt = os.getenv("FLUX_PROMPT")

    # Asegura padding correcto del Base64
    b64 = payload.imageBase64.strip()
    b64 += "=" * (-len(b64) % 4)
    img_bytes = base64.b64decode(b64)

    # Replicate necesita la imagen de entrada en Base64 con el prefijo de tipo de dato
    input_image_b64_prefixed = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

    try:
        # Llamada a la API de Replicate con el modelo y la versión correctos
        # y los parámetros de entrada validados
        output_url = replicate.run(
            "black-forest-labs/flux-kontext-pro:ks1w6tyk9nrma0cq6ycacv92xm", # Modelo y versión confirmados
            input={
                "input_image": input_image_b64_prefixed, # Clave correcta para la imagen de entrada
                "prompt": prompt,
                "negative_prompt": "cartoon, painting, illustration, low quality, bad quality, ugly, blurry, deformed",
                "aspect_ratio": "match_input_image", # Esto mantendrá las proporciones de tu imagen original
                "prompt_upsampling": False, # Puedes cambiar a True si quieres que el prompt se mejore automáticamente
                # 'guidance_scale' no está en la documentación de este modelo Pro, por lo que lo hemos eliminado.
                # 'safety_tolerance' por defecto es 2, no es necesario especificarlo a menos que quieras otro valor.
                # 'seed' puedes añadirlo si necesitas resultados reproducibles para pruebas
            }
        )
        
        # Replicate devuelve una URL temporal de la imagen generada.
        # Necesitamos descargarla y convertirla a Base64.
        response = requests.get(output_url)
        response.raise_for_status() # Lanza una excepción para códigos de estado HTTP de error
        modified_image_bytes = response.content

        mod_b64 = base64.b64encode(modified_image_bytes).decode("utf-8")
        return {"modifiedImage": mod_b64}

    except Exception as e:
        # Devuelve el detalle real del error para depuración
        print(f"Error procesando la imagen: {e}") # Para ver el error en tus logs
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen con Replicate: {str(e)}")
