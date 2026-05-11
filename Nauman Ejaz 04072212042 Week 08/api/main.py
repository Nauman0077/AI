from pathlib import Path
import io
import sys

import joblib
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from week7_extraction import extract_amounts, extract_dates, extract_entities

app = FastAPI(
    title='Document Intelligence API',
    description='OCR, classification, and information extraction',
    version='1.0.0',
)

vectorizer = joblib.load(ROOT_DIR / 'models' / 'vectorizer.pkl')
classifier = joblib.load(ROOT_DIR / 'models' / 'classifier.pkl')


def read_image(contents):
    return Image.open(io.BytesIO(contents)).convert('RGB')


def extract_text(image):
    return pytesseract.image_to_string(image)


def classify_text(text):
    text_vector = vectorizer.transform([text])
    prediction = classifier.predict(text_vector)[0]
    probabilities = classifier.predict_proba(text_vector)[0]
    return prediction, probabilities


@app.get('/')
def root():
    return {
        'message': 'Document Intelligence API',
        'version': '1.0.0',
        'endpoints': ['/classify', '/extract', '/process'],
    }


@app.post('/classify')
async def classify_document(file: UploadFile = File(...)):
    try:
        image = read_image(await file.read())
        text = extract_text(image)
        prediction, probabilities = classify_text(text)
        return {
            'document_type': prediction,
            'confidence': float(max(probabilities)),
            'all_probabilities': {
                label: float(probability)
                for label, probability in zip(classifier.classes_, probabilities)
            },
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={'error': str(exc)})


@app.post('/extract')
async def extract_information(file: UploadFile = File(...)):
    try:
        image = read_image(await file.read())
        text = extract_text(image)
        return {
            'dates': extract_dates(text),
            'amounts': extract_amounts(text),
            'entities': extract_entities(text),
            'raw_text': text[:500],
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={'error': str(exc)})


@app.post('/process')
async def process_document(file: UploadFile = File(...)):
    try:
        image = read_image(await file.read())
        text = extract_text(image)
        prediction, probabilities = classify_text(text)
        return {
            'document_type': prediction,
            'confidence': float(max(probabilities)),
            'extracted_data': {
                'dates': extract_dates(text),
                'amounts': extract_amounts(text),
                'entities': extract_entities(text),
            },
            'status': 'success',
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={'error': str(exc), 'status': 'failed'})
