# Lab 8: Document Classification and REST API

This submission contains two equivalent implementations of the same lab task.

- `version_1_functional`: function-based implementation
- `version_2_class_based`: class-based implementation

Both versions include:

- sample training images for invoices, receipts, and contracts
- TF-IDF document classifier training code
- saved `vectorizer.pkl` and `classifier.pkl` files
- FastAPI application
- `/classify`, `/extract`, and `/process` endpoints
- simple information extraction helpers
- notebook version of the classifier/API workflow

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Tesseract OCR must also be installed on the system for OCR to work.

## Train the model

From either version folder:

```bash
python train_classifier.py
```

The trained files are saved in `models/`.

## Run the API

From either version folder:

```bash
cd api
uvicorn main:app --reload
```

Open the Swagger test page:

```text
http://127.0.0.1:8000/docs
```

Use the provided sample images from `training_data/` to test each endpoint.
