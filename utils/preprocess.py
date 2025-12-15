import re

def clean_text(text):
    """
    Simple text cleaning: lowercases, removes special chars, and strips.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()