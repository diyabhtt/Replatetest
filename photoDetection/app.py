# ============================================================
# app.py — FastAPI Ingredient Detection API
# ============================================================

import os, shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from models.gemini_model import detect_ingredients

app = FastAPI(
    title="Ingredient Detection API",
    description="Upload an image → Gemini detects ingredients → Saves results locally + to Supabase."
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image file → run detection → return JSON.
    """
    os.makedirs("uploads", exist_ok=True)
    image_path = os.path.join("uploads", file.filename)

    # Save uploaded image
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run Gemini detection + Supabase sync
    data = detect_ingredients(image_path)

    return JSONResponse(content=data)

@app.get("/")
async def root():
    return {"message": "Welcome to the Ingredient Detection API. Use POST /detect to upload an image."}
