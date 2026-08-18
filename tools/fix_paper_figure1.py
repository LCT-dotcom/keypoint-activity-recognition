from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


PAGE_INDEX = 3


def _wrapped_lines(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _box(
    drawing: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
) -> None:
    drawing.setLineWidth(0.45)
    drawing.setStrokeColorRGB(0.35, 0.35, 0.35)
    drawing.setFillColorRGB(1, 1, 1)
    drawing.rect(x, y, width, height, stroke=1, fill=1)
    drawing.setFillColorRGB(0.08, 0.08, 0.08)
    drawing.setFont("Helvetica-Bold", 5.4)
    drawing.drawCentredString(x + width / 2, y + height - 8, title)
    drawing.setFont("Helvetica", 4.7)
    text_y = y + height - 15
    for line in _wrapped_lines(detail, 31):
        drawing.drawCentredString(x + width / 2, text_y, line)
        text_y -= 5.4


def _arrow(
    drawing: canvas.Canvas, x1: float, y1: float, x2: float, y2: float
) -> None:
    drawing.setStrokeColorRGB(0.25, 0.25, 0.25)
    drawing.setFillColorRGB(0.25, 0.25, 0.25)
    drawing.setLineWidth(0.55)
    drawing.line(x1, y1, x2, y2)
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        drawing.line(x2, y2, x2 - 3 * direction, y2 + 1.7)
        drawing.line(x2, y2, x2 - 3 * direction, y2 - 1.7)
    else:
        direction = 1 if y2 > y1 else -1
        drawing.line(x2, y2, x2 - 1.7, y2 - 3 * direction)
        drawing.line(x2, y2, x2 + 1.7, y2 - 3 * direction)


def build_overlay(width: float, height: float) -> PdfReader:
    stream = BytesIO()
    drawing = canvas.Canvas(stream, pagesize=(width, height))

    drawing.setFillColorRGB(1, 1, 1)
    drawing.rect(66, 608, 465, 142, stroke=0, fill=1)

    top_y, bottom_y = 681, 623
    box_w, box_h = 98, 45
    top_x = (77, 191, 305, 419)
    bottom_x = (134, 248, 362)

    top = (
        ("Raw 2D pose data", "17 COCO joints, four training participants, eight activities"),
        ("Clean and normalize", "interpolate missing joints, remove None for training, torso scale"),
        ("Sliding windows", "150 frames at 30 FPS; 50% training overlap"),
        ("Feature extraction", "TSFEL statistics plus velocity, acceleration, angles, distances"),
    )
    bottom = (
        ("LOSO evaluation", "held-out subject accuracy, macro F1, abnormal F1, confusion matrix"),
        ("HistGradientBoosting", "regularized multiclass classifier selected on training subjects"),
        ("Feature selection", "remove correlated features; fit selector on training data only"),
    )

    for x, (title, detail) in zip(top_x, top):
        _box(drawing, x, top_y, box_w, box_h, title, detail)
    for x, (title, detail) in zip(bottom_x, bottom):
        _box(drawing, x, bottom_y, box_w, box_h, title, detail)

    center_y = top_y + box_h / 2
    for left, right in zip(top_x, top_x[1:]):
        _arrow(drawing, left + box_w, center_y, right, center_y)
    _arrow(
        drawing,
        top_x[-1] + box_w / 2,
        top_y,
        bottom_x[-1] + box_w / 2,
        bottom_y + box_h,
    )
    bottom_center = bottom_y + box_h / 2
    _arrow(drawing, bottom_x[-1], bottom_center, bottom_x[1] + box_w, bottom_center)
    _arrow(drawing, bottom_x[1], bottom_center, bottom_x[0] + box_w, bottom_center)

    drawing.save()
    stream.seek(0)
    return PdfReader(stream)


def correct_figure(input_path: Path, output_path: Path) -> Path:
    reader = PdfReader(input_path)
    if len(reader.pages) <= PAGE_INDEX:
        raise ValueError("The source PDF does not contain Figure 1 on page 4")
    page = reader.pages[PAGE_INDEX]
    overlay = build_overlay(float(page.mediabox.width), float(page.mediabox.height))
    page.merge_page(overlay.pages[0], over=True)

    writer = PdfWriter()
    for source_page in reader.pages:
        writer.add_page(source_page)
    if reader.metadata:
        writer.add_metadata(reader.metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        writer.write(stream)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an author-corrected copy with a non-duplicated Figure 1."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(correct_figure(args.input, args.output).resolve())
