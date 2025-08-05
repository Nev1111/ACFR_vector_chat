# extract_pdf.py
from pypdf import PdfReader

def extract_text_from_pdf(path: str) -> str:
    """
    Read a PDF and return all pages' text as one long string.
    """
    reader = PdfReader(path)
    pages_text = [page.extract_text() for page in reader.pages]
    return "\n".join(pages_text)
