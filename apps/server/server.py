"""
nano-vllm server module: FastAPI-based inference server for nano-vllm LLM engine.
Similar in spirit to llama-cpp's llama-server, but adapted for nano-vllm.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional

import uvicorn
import logging
import os

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams
import threading

app = FastAPI(title="nano-vllm Server", description="FastAPI server for nano-vllm inference.")

logger = logging.getLogger("nano-vllm-server")
logging.basicConfig(level=logging.INFO)

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    repetition_penalty: float = 1.0
    profile: Optional[bool] = False  # If True, run cProfile and log results

class GenerationResponse(BaseModel):
    generated_text: str
    # Add more fields as needed (e.g., token usage, timings)


# --- OpenAI-Compatible Endpoint Models ---
class OpenAIChatMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = None
    repetition_penalty: Optional[float] = 1.0

class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: Optional[str] = None

class OpenAIChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatChoice]

# --- Engine Initialization ---
engine = None
engine_lock = threading.Lock()

class EngineArgs:
    # Minimal args for demo; expand as needed
    def __init__(self, model: Optional[str] = None, use_mps: bool = True):
        # Allow model path to be set via environment variable
        self.model = model or os.environ.get("NANO_VLLM_MODEL_PATH", "Qwen/Qwen1.5-0.5B")
        self.use_mps = use_mps

def get_engine():
    global engine
    with engine_lock:
        if engine is None:
            logger.info("Loading LLMEngine...")
            args = EngineArgs()  # Optionally, make model path configurable
            engine_instance = LLMEngine.from_engine_args(args)
            engine = engine_instance
            logger.info("LLMEngine loaded.")
        return engine

@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    logger.info(f"Received generation request: {request}")
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    if request.profile:
        import cProfile
        import pstats
        import io
        pr = cProfile.Profile()
        pr.enable()
        outputs = engine.generate([request.prompt], sampling_params, use_tqdm=False)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Top 20
        logger.info("\n[PROFILE] Top 20 cumulative time functions:\n" + s.getvalue())
    else:
        outputs = engine.generate([request.prompt], sampling_params, use_tqdm=False)
    generated_text = outputs[0]["text"] if outputs and isinstance(outputs[0]["text"], str) else ""
    return GenerationResponse(generated_text=generated_text)


# --- OpenAI-Compatible Chat Completion Endpoint ---
@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
def openai_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    logger.info(f"Received OpenAI chat completion request: {request}")
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")
    # Compose prompt from messages (OpenAI format)
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    sampling_params = SamplingParams(
        temperature=request.temperature if request.temperature is not None else 1.0,
        max_tokens=request.max_tokens if request.max_tokens is not None else 128,
    )
    outputs = engine.generate([prompt], sampling_params, use_tqdm=False)
    generated_text = outputs[0]["text"] if outputs and isinstance(outputs[0]["text"], str) else ""
    # Return as OpenAI-compatible response
    response = OpenAIChatResponse(
        id="chatcmpl-1",
        object="chat.completion",
        created=int(__import__("time").time()),
        model=request.model,
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatMessage(role="assistant", content=generated_text),
                finish_reason="stop",
            )
        ],
    )
    return response

if __name__ == "__main__":
    uvicorn.run("apps.server.server:app", host="0.0.0.0", port=8000, reload=True)
