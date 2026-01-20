import cv2
from fastapi import HTTPException, UploadFile
import pytesseract
import numpy as np
from PIL import Image
import arabic_reshaper
from bidi.algorithm import get_display

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

def ocr_with_confidence(image):
    data = pytesseract.image_to_data(
        image,
        config='-l ara --psm 7',
        output_type=pytesseract.Output.DATAFRAME
    )

    data = data[data.conf > 60]
    return ' '.join(data.text.dropna())

def postprocess_arabic_text(text):
    # Reshape Arabic letters
    reshaped_text = arabic_reshaper.reshape(text)

    # Fix RTL direction
    bidi_text = get_display(reshaped_text)

    return bidi_text

def arabic_ocr_pipeline(file: UploadFile):
    '''This is main method from OCR Phase, returning from Image to text'''
    img = read_image_from_upload(file)
    processed_image = preprocess_image(img)

    raw_text = ocr_with_confidence(processed_image)

    # final_text = postprocess_arabic_text(raw_text)

    return raw_text

def read_image_from_upload(file: UploadFile):
    image_bytes = file.file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    return img


