#!/usr/bin/env python
"""Prueba simple del servidor"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from app.services.inference import get_inference_engine

app = FastAPI(title="PhytoLens Demo")

@app.get("/")
def root():
    return {"message": "PhytoLens IA Online"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    engine = get_inference_engine()
    return engine.get_model_info()

if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor en puerto 8001...")
    uvicorn.run(app, host="127.0.0.1", port=8001)
