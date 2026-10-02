"""Portfolio reconstruction of the recovered review preprocessing workflow.
The original student script is preserved separately in original_preprocessing.py.
Unlike the original corpus-wide top-frequency filters, the model pipeline learns
CountVectorizer features *inside* the training split to avoid evaluation leakage.
"""
import re

PATTERN = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

def normalize_review(text):
    """Use the recovered assignment's lowercase and regex-tokenization pattern."""
    tokens = PATTERN.findall(str(text if text is not None else '').lower())
    return ' '.join(t for t in tokens if len(t) > 1)
