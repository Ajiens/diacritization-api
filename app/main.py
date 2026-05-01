import base64

from fastapi import FastAPI, File, HTTPException, UploadFile
# from services import process_text
from app.ocr import arabic_ocr_pipeline
from app.addabit_diacritic import diacritic_text as addabit
from app.shakkala_diacritic import diacritic_text as shakkala
from app.request_model import DiacritizeRequest, OCRRequest

app = FastAPI(
    title="Simple NLP API",
    description="API sederhana berbasis FastAPI",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/diacritize")
async def diacritize_endpoint(req: DiacritizeRequest):
    if req.model == "ad-dabit":
        result = addabit(req.text)
    elif req.model == "shakkala":
        result = shakkala(req.text)
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified")

    return {
        "text": req.text,
        "model": req.model,
        "result": result
    }


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    model: str = "default"
):
    contents = await file.read()

    text = arabic_ocr_pipeline(contents)

    return {"result": text}

# async def arabic_ocr_endpoint(file: UploadFile = File(...)):
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File must be an image")

#     text = arabic_ocr_pipeline(file) #Get the text result
#     diacritic = diacritic_text(text)


#     return {
#         "text": text,
#         "diacritic": diacritic
#     }

