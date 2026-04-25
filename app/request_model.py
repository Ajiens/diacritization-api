from pydantic import BaseModel

class DiacritizeRequest(BaseModel):
    text: str
    model: str

class OCRRequest(BaseModel):
    file: str # Body are byte in format str
    model: str