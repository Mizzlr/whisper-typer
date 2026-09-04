#!/usr/bin/env python3
"""Resident Granite Speech 5 TurboCTC HTTP service for White Wolf."""

from contextlib import asynccontextmanager
from io import BytesIO
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor

MODEL_ID = "ibm-granite/granite-speech-5.0-470m-turboctc"


@asynccontextmanager
async def lifespan(app: FastAPI):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    app.state.processor = processor
    app.state.model = model

    # Allocate CUDA kernels and buffers before the first real dictation.
    silence = np.zeros(16_000, dtype=np.float32)
    inputs = processor(
        [silence],
        sampling_rate=processor.feature_extractor.sampling_rate,
        device=model.device,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        model.generate(**inputs)
    torch.cuda.synchronize()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ready", "model": MODEL_ID}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    payload = await audio.read()
    if len(payload) > 32 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio exceeds 32 MiB")
    try:
        samples, sample_rate = sf.read(BytesIO(payload), dtype="float32")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"invalid audio: {error}") from error
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    target_rate = app.state.processor.feature_extractor.sampling_rate
    if sample_rate != target_rate:
        raise HTTPException(
            status_code=400,
            detail=f"expected {target_rate} Hz audio, received {sample_rate} Hz",
        )

    started = time.perf_counter()
    inputs = app.state.processor(
        [samples],
        sampling_rate=sample_rate,
        device=app.state.model.device,
        return_tensors="pt",
    ).to(app.state.model.device, dtype=app.state.model.dtype)
    with torch.inference_mode():
        tokens = app.state.model.generate(**inputs)
    torch.cuda.synchronize()
    text = app.state.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    return {
        "text": text,
        "duration_s": len(samples) / sample_rate,
        "latency_ms": (time.perf_counter() - started) * 1000,
    }
