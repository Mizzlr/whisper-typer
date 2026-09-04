#!/usr/bin/env python3
"""Resident CUDA punctuation and truecasing service for Black Beast."""

from contextlib import asynccontextmanager
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from punctuators.models import PunctCapSegModelONNX

MODEL_ID = "1-800-BAD-CODE/punctuation_fullstop_truecase_english"


class PunctuationRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = PunctCapSegModelONNX.from_pretrained("pcs_en")
    providers = model._ort_session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA punctuation provider unavailable: {providers}")
    model.infer(["warm up the punctuation model now"])
    app.state.model = model
    app.state.lock = threading.Lock()
    app.state.providers = providers
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ready",
        "model": MODEL_ID,
        "providers": app.state.providers,
    }


@app.post("/punctuate")
def punctuate(request: PunctuationRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")
    if len(text) > 32_000:
        raise HTTPException(status_code=413, detail="text exceeds 32000 characters")
    started = time.perf_counter()
    with app.state.lock:
        sentences = app.state.model.infer([text])[0]
    corrected = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
    if not corrected:
        raise HTTPException(status_code=500, detail="model returned empty text")
    return {
        "text": corrected,
        "latency_ms": (time.perf_counter() - started) * 1000,
    }
