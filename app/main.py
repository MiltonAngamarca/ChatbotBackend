import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "ai/qwen3:0.6B-F16")
MODEL_RUNNER_URL = os.getenv(
    "MODEL_RUNNER_URL",
    "http://model-runner.docker.internal/engines/v1/chat/completions"
)

app = FastAPI(title="GEMMA AI API", version="1.0.0")

# CORS Configuration - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str

def remove_think_tags(text: str) -> str:
    """
    Elimina todo el contenido dentro de las etiquetas <think></think>
    """
    # Patrón regex para encontrar y eliminar contenido entre <think> y </think>
    # re.DOTALL hace que . coincida también con saltos de línea
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Limpiar espacios en blanco extra y saltos de línea múltiples
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Múltiples líneas vacías a máximo 2
    cleaned_text = cleaned_text.strip()  # Eliminar espacios al inicio y final
    
    return cleaned_text

@app.get("/")
async def root():
    return {"message": "DeepSeek AI API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME}

@app.options("/predict")
async def predict_options():
    """Handle preflight OPTIONS request"""
    return {"message": "OK"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        logger.info(f"Received prediction request: {req.text[:50]}...")
        
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": req.text}],
            "stream": False
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.info(f"Sending request to model runner: {MODEL_RUNNER_URL}")
            r = await client.post(MODEL_RUNNER_URL, json=payload)

        if r.status_code != 200:
            logger.error(f"Model runner error: {r.status_code} - {r.text}")
            raise HTTPException(r.status_code, f"Model runner error: {r.text}")

        data = r.json()
        logger.info("Received response from model runner")
        
        try:
            raw_content = data["choices"][0]["message"]["content"]
            
            # Eliminar las etiquetas <think></think> y su contenido
            cleaned_content = remove_think_tags(raw_content)
            
            logger.info(f"Raw response: {raw_content[:100]}...")
            logger.info(f"Cleaned response: {cleaned_content[:100]}...")
            
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format: {data}")
            raise HTTPException(500, f"Unexpected response format: {str(e)}")

        return {"prediction": cleaned_content}
        
    except httpx.TimeoutException:
        logger.error("Request to model runner timed out")
        raise HTTPException(504, "Request timed out")
    except httpx.RequestError as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(503, f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )