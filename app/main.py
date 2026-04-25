import base64

from fastapi import FastAPI, File, HTTPException, UploadFile
# from services import process_text
from ocr import arabic_ocr_pipeline
from diacritic import diacritic_text
from request_model import DiacritizeRequest, OCRRequest

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
    result = diacritic_text(req.text)

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

