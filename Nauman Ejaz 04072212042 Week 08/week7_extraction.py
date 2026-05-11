import re


def extract_dates(text):
    patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b'
    ]
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))
    return list(dict.fromkeys(dates))


def extract_amounts(text):
    patterns = [
        r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?',
        r'\b\d+(?:,\d{3})*\.\d{2}\b'
    ]
    amounts = []
    for pattern in patterns:
        amounts.extend(re.findall(pattern, text))
    return list(dict.fromkeys(amounts))


def extract_entities(text):
    candidates = re.findall(r'\b[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+){0,3}\b', text)
    return list(dict.fromkeys(candidates[:20]))
