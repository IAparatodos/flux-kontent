# backend.py (FastAPI)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, base64, io

# Importa el cliente de Replicate
import replicate

load_dotenv()  # carga REPLICATE_API_TOKEN y FLUX_PROMPT desde .env o variables de entorno

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tudominio.com"],  # ajusta al dominio de tu WordPress
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Inicializa el cliente de Replicate con tu token
# Replicate busca REPLICATE_API_TOKEN automáticamente en las variables de entorno
# Puedes configurarlo en tu .env o en el entorno de Digital Ocean
# replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

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
    # o una URL pública si no quieres Base64 directamente.
    # Para simplicidad y porque ya tienes los bytes, convertimos a Base64 con el prefijo.
    input_image_b64_prefixed = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

    try:
        # Llamada a la API de Replicate
        # El modelo completo en Replicate es "black-forest-labs/flux-kontext-dev"
        # y la tarea image-to-image se maneja pasando la imagen de entrada.
        # Los parámetros exactos pueden variar ligeramente, consulta la página del modelo en Replicate.
        # Por ejemplo, para flux-kontext-dev, la entrada es 'image' y 'prompt'.

        output_url = replicate.run(
            "black-forest-labs/flux-kontext-dev:e872c366ff2f1f513d28905322a36b99092497e2316e11802bb0ff0970a0494b", # Versión específica del modelo
            input={
                "image": input_image_b64_prefixed,
                "prompt": prompt,
                "negative_prompt": "cartoon, painting, illustration, low quality, bad quality, ugly, blurry, deformed", # Ejemplo de prompt negativo, revisa la doc de Replicate para este modelo
                "guidance_scale": 2.5
                # Aquí puedes añadir más parámetros según la documentación de Replicate para el modelo
            }
        )
        # Replicate devuelve una URL temporal de la imagen generada.
        # Necesitamos descargarla y convertirla a Base64.
        import requests
        response = requests.get(output_url)
        response.raise_for_status() # Asegura que la descarga fue exitosa
        modified_image_bytes = response.content

        mod_b64 = base64.b64encode(modified_image_bytes).decode("utf-8")
        return {"modifiedImage": mod_b64}

    except Exception as e:
        # Devuelve el detalle real del error para depuración
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen con Replicate: {str(e)}")
