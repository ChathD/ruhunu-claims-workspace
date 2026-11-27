"""Extract searchable wording and run local OCR on scanned policy pages."""
import re
import sys
import threading
from bisect import bisect_right
from contextlib import closing
from functools import lru_cache

PDF_LOCK = threading.Lock()
LIMIT_CELL = re.compile(r"(?:unlimited|\d+(?:\.\d+)?\s*%|(?:lkr|rs\.?)\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?(?:/-)?)", re.I)
OCR_CURRENCY = re.compile(r"(?:\blkr|\brs\.?|rs\.)\s*(?P<amount>[0-9o]+(?:[ \t]*,[ \t]*[0-9o]+)*(?:\.[0-9o]+)?)", re.I)


@lru_cache(maxsize=1)
def ocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)


def windows_scan_lines(page, scale):
    from pypdfium2.raw import FPDFBitmap_BGRA
    from winrt.runtime import ApartmentType, init_apartment, uninit_apartment
    from winrt.windows.foundation import IAsyncOperation  # Register the projected async return type.
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import Buffer

    init_apartment(ApartmentType.MULTI_THREADED)
    try:
        engine = OcrEngine.try_create_from_language(Language("en"))
        if engine is None:
            return None
        with closing(page.render(scale=scale, force_bitmap_format=FPDFBitmap_BGRA)) as bitmap:
            pixels = bytes(bitmap.buffer)
            buffer = Buffer(len(pixels))
            buffer.length = len(pixels)
            memoryview(buffer)[:] = pixels
            image = SoftwareBitmap.create_copy_from_buffer(buffer, BitmapPixelFormat.BGRA8,
                                                             bitmap.width, bitmap.height, BitmapAlphaMode.IGNORE)
            with closing(image):
                operation: IAsyncOperation = engine.recognize_async(image)
                result = operation.get()
            lines = []
            for line in result.lines:
                bounds = [word.bounding_rect for word in line.words]
                if bounds:
                    lines.append((min(rect.y for rect in bounds), min(rect.x for rect in bounds),
                                  max(rect.x + rect.width for rect in bounds), line.text))
            return bitmap.width, lines
    finally:
        uninit_apartment()


def portable_scan_lines(page, scale, *, crop=(0, 0, 0, 0)):
    with closing(page.render(scale=scale, crop=crop)) as bitmap:
        lines, _ = ocr_engine()(bitmap.to_numpy(), use_cls=False)
        return bitmap.width, [(box[0][1], min(x for x, _ in box), max(x for x, _ in box), text)
                              for box, text, _ in lines or []]


def single_column_scan(page, scale, width, positioned, native):
    tolerance = width * 0.005
    margin = width * 0.75
    portable = None
    if native and any(suspect_currency(line[3]) for line in positioned):
        _, portable = portable_scan_lines(page, scale)
        positioned = reconcile_currency(positioned, portable, tolerance)
    anchors = [y for y, x, _, text in positioned if x >= margin and LIMIT_CELL.fullmatch(text.strip())]
    if native and anchors:
        # The native recognizer can miss isolated values at the right of a liability table.
        if portable is None:
            crop_width, values = portable_scan_lines(page, scale, crop=(page.get_width() * 0.75, 0, 0, 0))
            offset = width - crop_width
        else:
            values = [line for line in portable if line[1] >= margin]
            offset = 0
        for y, start, end, text in values:
            if (not LIMIT_CELL.fullmatch(text.strip()) or not any(abs(y - anchor) < width * 0.06 for anchor in anchors)
                    or not any(abs(y - line_y) <= tolerance and line_end < offset + start
                               for line_y, _, line_end, _ in positioned)
                    or any(abs(y - line_y) <= tolerance and line_end > offset + start
                           for line_y, _, line_end, _ in positioned)):
                continue
            positioned.append((y, offset + start, offset + end, text))

    rows = []
    for line in sorted(positioned):
        if not rows or line[0] - rows[-1][0][0] > tolerance:
            rows.append([])
        rows[-1].append(line)
    return "\n".join("  ".join(line[3] for line in sorted(row, key=lambda line: line[1])) for row in rows)


def suspect_currency(text):
    match = OCR_CURRENCY.search(text)
    if match and (re.fullmatch(r"[1-9]", match["amount"]) or
                  ("o" in match["amount"].lower() and re.search(r"\d", match["amount"]))):
        return match
    return None


def reconcile_currency(native, portable, tolerance):
    def context(text, match):
        return re.sub(r"[^a-z0-9]", "", (text[:match.start("amount")] + text[match.end("amount"):]).casefold())

    repaired = []
    for y, x, end, text in native:
        match = suspect_currency(text)
        if match:
            observed = re.sub(r"\s", "", match["amount"].lower().replace("o", "0"))
            matches = set()
            for other_y, other_x, other_end, other in portable:
                if abs(y - other_y) > tolerance or other_end <= x or end <= other_x:
                    continue
                candidate = OCR_CURRENCY.search(other)
                if candidate and context(text, match) == context(other, candidate):
                    amount = re.sub(r"\s", "", candidate["amount"].lower().replace("o", "0"))
                    agrees = amount == observed or (len(observed) == 1 and amount.startswith(observed))
                    if agrees and re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?", amount):
                        matches.add(amount)
            if len(matches) == 1:
                text = text[:match.start("amount")] + matches.pop() + text[match.end("amount"):]
        repaired.append((y, x, end, text))
    return repaired


def ruled_grids(dark):
    import numpy as np

    width = dark.shape[1]
    tolerance = width * 0.004
    groups = []
    ys, xs = np.nonzero(np.diff(np.pad(dark, ((0, 0), (1, 1))), axis=1))
    for y, left, right in zip(ys[::2], xs[::2], xs[1::2], strict=True):
        if not width * 0.15 <= right - left < width * 0.98:
            continue
        for start, end, rules in groups:
            if abs(left - start) <= tolerance and abs(right - end) <= tolerance:
                if y - rules[-1] > tolerance:
                    rules.append(y)
                break
        else:
            groups.append((left, right, [y]))

    for left, right, rules in groups:
        for index, top in enumerate(rules[:-2]):
            bottom = rules[-1]
            if bottom - top < width * 0.04:
                break
            vertical = dark[top:bottom, left:right].mean(axis=0) >= 0.8
            starts = np.flatnonzero(np.diff(np.r_[False, vertical, False]))[::2] + left
            columns = [int(x) for x in starts if x - left > 2 * tolerance and right - x > 2 * tolerance]
            if columns:
                yield rules[index:], [left, *columns, right]
                break


def preserve_table_rows(page, scale, positioned):
    with closing(page.render(scale=scale, grayscale=True)) as bitmap:
        grids = list(ruled_grids(bitmap.to_numpy().squeeze() < 125))
    for rows, columns in grids:
        matched, cells = [], {}
        for line in positioned:
            y, left, right, text = line
            if not (rows[0] <= y < rows[-1] and columns[0] <= left < right <= columns[-1]):
                continue
            column = bisect_right(columns, left) - 1
            if right > columns[column + 1]:
                break  # The OCR box spans cells. Keep the original extraction rather than guessing.
            matched.append(line)
            row = bisect_right(rows, y) - 1
            cells.setdefault((row, column), []).append(line)
        else:
            if len({column for _, column in cells}) < 2:
                continue
            table_rows = []
            for index, y in enumerate(rows[:-1]):
                values = [" ".join(line[3] for line in sorted(cells.get((index, column), [])))
                          for column in range(len(columns) - 1)]
                if any(values):
                    table_rows.append((y, columns[0], columns[-1], " | ".join(values)))
            positioned = [line for line in positioned if line not in matched] + table_rows
    return positioned


def read_scan(page):
    scale = min(2.8, 2400 / max(page.get_size()))
    lines = None
    if sys.platform == "win32":
        try:
            lines = windows_scan_lines(page, scale)
        except (ImportError, OSError):
            # The native recognizer may be unavailable on a Windows installation.
            pass
    width, positioned = lines if lines is not None else portable_scan_lines(page, scale)
    positioned = preserve_table_rows(page, scale, positioned)
    midpoint = width / 2
    left = [line for line in positioned if line[2] <= midpoint and line[2] - line[1] > midpoint * 0.3]
    right = [line for line in positioned if line[1] >= midpoint and line[2] - line[1] > midpoint * 0.3]
    if len(left) < 8 or len(right) < 8 or abs(left[0][0] - right[0][0]) > 300:
        return single_column_scan(page, scale, width, positioned, native=lines is not None)

    ordered, band = [], []
    for line in sorted(positioned):
        if line[1] < midpoint < line[2]:
            ordered.extend(sorted(band, key=lambda row: (row[1] >= midpoint, row[0], row[1])))
            band.clear()
            ordered.append(line)
        else:
            band.append(line)
    ordered.extend(sorted(band, key=lambda row: (row[1] >= midpoint, row[0], row[1])))
    return "\n".join(line[3] for line in ordered)


def extract_pdf_pages(raw):
    from pypdfium2 import PdfDocument, PdfiumError

    try:
        # PDFium handles must not overlap across threads.
        with PDF_LOCK, PdfDocument(raw) as pdf:
            pages, scanned = [], []
            for index in range(len(pdf)):
                with closing(pdf[index]) as page, closing(page.get_textpage()) as pdf_text:
                    text = pdf_text.get_text_bounded().replace("\r\n", "\n").strip()
                    if not text or re.fullmatch(r"(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?", text, re.I):
                        text = read_scan(page)
                        scanned.append(index + 1)
                    if not text.strip():
                        raise ValueError(f"OCR could not read policy page {index + 1}. Upload a clearer PDF.")
                    pages.append((index + 1, text))
            return pages, scanned
    except PdfiumError as error:
        raise ValueError("The policy PDF could not be read. Check that it opens and is unlocked.") from error
