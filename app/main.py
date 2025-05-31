import os
import re
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    stream: bool = False

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

async def generate_streaming_response(req: PredictRequest):
    """Generator function for streaming responses - simplified to pass through model runner chunks"""
    stream_ended_by_model = False
    try:
        logger.info(f"Received streaming prediction request: {req.text[:50]}...")
        
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": req.text}],
            "stream": True
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.info(f"Sending streaming request to model runner: {MODEL_RUNNER_URL}")
            async with client.stream('POST', MODEL_RUNNER_URL, json=payload) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    error_detail = error_content.decode(errors='ignore')
                    logger.error(f"Model runner error: {response.status_code} - {error_detail}")
                    error_event = {"error": f"Model runner error: {response.status_code}", "details": error_detail}
                    yield f"data: {json.dumps(error_event)}\n\n"
                    # No return here, finally block will send [DONE]
                else:
                    logger.info("Successfully connected to model runner for streaming.")
                    async for chunk in response.aiter_lines(): # Process line by line
                        # Model runner (OpenAI compatible) should send SSE formatted lines:
                        # e.g., data: {"id": ..., "choices": [{"delta": {"content": "..."}}]}
                        # or   data: [DONE]
                        logger.debug(f"Raw line from model runner: '{chunk}'")
                        if chunk.strip() == "data: [DONE]":
                            logger.info("Model runner sent 'data: [DONE]'. Forwarding and marking stream end.")
                            yield "data: [DONE]\n\n" # Forward the [DONE] signal
                            stream_ended_by_model = True
                            return # Exit the generator as stream is complete
                        elif chunk.startswith("data:"):
                            yield f"{chunk}\n\n" # Forward the data line, ensuring it ends with double newline for SSE
                        elif chunk: # Non-empty line that isn't a data line (e.g. comments, keep-alive pings)
                            logger.info(f"Received non-data line from model runner: '{chunk}'. Forwarding as comment or ignoring.")
                            # Optionally, forward as SSE comment: yield f": {chunk}\n\n"
                            # For now, just log and don't forward to keep it clean unless it's data.
                    logger.info("Finished iterating over model runner response chunks.")

    except httpx.TimeoutException:
        logger.error("Streaming request to model runner timed out")
        error_event = {"error": "Request to model runner timed out"}
        yield f"data: {json.dumps(error_event)}\n\n"
    except httpx.RequestError as e:
        logger.error(f"Streaming request error: {str(e)}")
        error_event = {"error": f"Service unavailable: {str(e)}"}
        yield f"data: {json.dumps(error_event)}\n\n"
    except Exception as e:
        logger.error(f"Unexpected streaming error: {str(e)}", exc_info=True)
        error_event = {"error": f"Internal server error: {str(e)}"}
        yield f"data: {json.dumps(error_event)}\n\n"
    finally:
        if not stream_ended_by_model:
            logger.info("Stream not ended by model runner or error occurred. Ensuring [DONE] signal is sent to client.")
            yield "data: [DONE]\n\n"
        else:
            logger.info("Stream already ended by model runner with [DONE]. No additional [DONE] from finally block.")

@app.post("/predict")
async def predict(req: PredictRequest):
    logger.info(f"Solicitud recibida para /predict. Raw request object type: {type(req)}, stream value: {req.stream}, text: {req.text[:50]}...")
    if req.stream:
        return StreamingResponse(
            generate_streaming_response(req),
            media_type="text/event-stream", # Correct media type for SSE
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # "Content-Type" is set by StreamingResponse based on media_type
            }
        )
    else:
        # Non-streaming response (original behavior)
        try:
            logger.info(f"Received non-streaming prediction request: {req.text[:50]}...")
            
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": req.text}],
                "stream": False
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                logger.info(f"Sending non-streaming request to model runner: {MODEL_RUNNER_URL}")
                r = await client.post(MODEL_RUNNER_URL, json=payload)

            r.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses

            data = r.json()
            logger.info("Received non-streaming response from model runner")
            
            try:
                raw_content = data["choices"][0]["message"]["content"]
                cleaned_content = remove_think_tags(raw_content)
                logger.info(f"Cleaned non-streaming response: {cleaned_content[:100]}...")
                return {"prediction": cleaned_content}
                
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected non-streaming response format: {data}", exc_info=True)
                raise HTTPException(500, f"Unexpected response format from model: {str(e)}")

        except httpx.TimeoutException:
            logger.error("Non-streaming request to model runner timed out")
            raise HTTPException(504, "Request to model runner timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"Model runner HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(e.response.status_code, f"Model runner error: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Non-streaming request error: {str(e)}", exc_info=True)
            raise HTTPException(503, f"Service unavailable: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected non-streaming error: {str(e)}", exc_info=True)
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