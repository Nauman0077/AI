import os
from pathlib import Path

import joblib
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def image_to_text(path):
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception:
        return ''


def load_documents(data_dir):
    documents, labels = [], []
    for doc_type in sorted(os.listdir(data_dir)):
        folder_path = Path(data_dir) / doc_type
        if not folder_path.is_dir():
            continue
        for file_path in sorted(folder_path.iterdir()):
            if file_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}:
                continue
            text = image_to_text(file_path)
            if text.strip():
                documents.append(text)
                labels.append(doc_type)
    return documents, labels


def train_classifier(documents, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        documents,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_vec, y_train)
    predictions = classifier.predict(X_test_vec)
    print(f'Accuracy: {accuracy_score(y_test, predictions):.2%}')
    print(classification_report(y_test, predictions))
    return vectorizer, classifier


def save_models(vectorizer, classifier, model_dir='models'):
    Path(model_dir).mkdir(exist_ok=True)
    joblib.dump(vectorizer, Path(model_dir) / 'vectorizer.pkl')
    joblib.dump(classifier, Path(model_dir) / 'classifier.pkl')


if __name__ == '__main__':
    docs, labels = load_documents('training_data')
    if len(docs) < 6:
        raise ValueError('Add training images or confirm Tesseract OCR is installed.')
    vec, clf = train_classifier(docs, labels)
    save_models(vec, clf)
    print('Models saved successfully.')
