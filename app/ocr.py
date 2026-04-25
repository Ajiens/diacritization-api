import cv2
from fastapi import HTTPException, UploadFile
import pytesseract
import numpy as np
from PIL import Image

def preprocess_image(img):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoising
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Threshold (baik untuk teks Arab)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    # Morphological operation (opsional)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return processed

def do_tesseract(image):
    # Sesuaikan config jika perlu
    custom_config = f'--oem 1 --psm 3'
    text = pytesseract.image_to_string(image, lang='ara', config=custom_config)
    return text

def arabic_ocr_pipeline(file: bytes) -> str:
    '''This is main method from OCR Phase, returning from Image to text'''
    img = read_image_from_bytes(file)
    processed_image = preprocess_image(img)

    raw_text = do_tesseract(processed_image)

    return raw_text

def read_image_from_bytes(image_bytes: bytes):
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    return img


