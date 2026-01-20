from fastapi import FastAPI, File, HTTPException, UploadFile
from app.services import process_text
from app.ocr import arabic_ocr_pipeline
from app.diacrtic import diacritic_text

app = FastAPI(
    title="Simple NLP API",
    description="API sederhana berbasis FastAPI",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/diacritic")
async def arabic_ocr_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    text = arabic_ocr_pipeline(file) #Get the text result
    diacritic = diacritic_text(text)


    return {
        "text": text,
        "diacritic": diacritic
    }

