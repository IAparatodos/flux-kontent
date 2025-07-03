# backend.py (FastAPI)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os, base64

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tudominio.com"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

hf = InferenceClient(token=os.getenv('HF_TOKEN'))

class Payload(BaseModel):
    imageBase64: str

@app.post('/api/editar-imagen')
async def editar(payload: Payload):
    prompt = os.getenv('FLUX_PROMPT')
    b64 = payload.imageBase64.strip()
    # añade los '=' que falten para completar a múltiplo de 4
    b64 += "=" * (-len(b64) % 4)
    img_bytes = base64.b64decode(b64)

    try:
        result = hf.image_editing(
            model='black-forest-labs/flux-kontext-dev',
            inputs=img_bytes,
            parameters={'prompt': prompt, 'guidance_scale': 2.5}
        )
        img_out = result[0]['generated_image']
        return {'modifiedImage': img_out}
   except Exception as e:
        # Con esto verás el error real en Swagger y en los logs
        raise HTTPException(status_code=500, detail=str(e))
