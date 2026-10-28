"""
FastAPI app entrypoint.

Routes are intentionally thin — all actual logic lives in
predictor.py / core/. This file just wires HTTP in and out.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import SkinLesionPredictor

app = FastAPI(title="Skin Lesion Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = SkinLesionPredictor()


@app.get("/")
def root():
    return {"status": "ok", "message": "Skin Lesion Detection API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "device": str(predictor.device)}


@app.get("/model-info")
def model_info():
    return predictor.model_metadata


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age: float = Form(...),
    sex: str = Form(...),
    localization: str = Form(...),
):
    try:
        image_bytes = await image.read()
        result = predictor.predict(image_bytes, age, sex, localization)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/reload-model")
def reload_model(model_filename: str = Form(...)):
    try:
        predictor.reload(model_filename=model_filename)
        return JSONResponse({"status": "reloaded", "model_info": predictor.model_metadata})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
