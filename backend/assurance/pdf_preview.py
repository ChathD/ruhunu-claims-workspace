"""Create readable page images and accessible text from a generated PDF."""
import base64
from contextlib import closing
from io import BytesIO

from pypdfium2 import PdfDocument

from backend.evidence.policy_pdf import PDF_LOCK


def pdf_page_previews(content):
    pages = []
    with PDF_LOCK, PdfDocument(content) as pdf:
        for index in range(len(pdf)):
            with closing(pdf[index]) as page, closing(page.render(scale=1.65)) as bitmap, closing(page.get_textpage()) as text_page:
                image = BytesIO()
                bitmap.to_pil().save(image, format="PNG")
                pages.append({"png": base64.b64encode(image.getvalue()).decode("ascii"), "text": text_page.get_text_bounded()})
    return pages
