#!/usr/bin/env python3
"""Generate synthetic test images with EAN-13 barcodes for retry scenarios.

Creates high-quality images in samples/synthetic/ that the zxing scanner can
read reliably. Barcodes are rendered at high resolution with crisp edges.

Images:
  - photo1_12_9found.png: 12 labels, 9 scannable, 3 blurred (missing)
  - retry_exact_3.png: 3 labels with the 3 missing barcodes (NEW barcodes)
  - retry_less_1.png: 1 label with 1 missing barcode (NEW barcode)
  - retry_more_5.png: 5 labels — 3 new + 2 from photo1 (neighbors)
  - agg_photo1_6.png: 6 labels, all scannable (complete)
  - agg_photo2_5.png: 5 labels with NEW barcodes (aggregation)
"""
from pathlib import Path

from barcode import get_barcode_class
from barcode.writer import ImageWriter
from PIL import Image, ImageFilter

OUT = Path("samples/synthetic")
OUT.mkdir(parents=True, exist_ok=True)

# Use Code128 to encode 13-digit EAN-13 values. The scanner reads Code128,
# and the PrimaryShoeboxBarcodePolicy accepts 13-digit values with valid
# EAN-13 checksums regardless of the barcode format.
CODE128 = get_barcode_class("code128")

# 12 distinct EAN-13 barcodes for photo1 (valid checksums)
PHOTO1_BARCODES = [
    "7290001000014",
    "7290001000021",
    "7290001000038",
    "7290001000045",
    "7290001000052",
    "7290001000069",
    "7290001000076",
    "7290001000083",
    "7290001000090",
    # these 3 will be blurred (missing in photo1)
    "7290001000106",
    "7290001000113",
    "7290001000120",
]

# 3 NEW barcodes for retry_exact (the 3 missing boxes, but with detectable barcodes)
RETRY_EXACT_BARCODES = [
    "7290001000106",
    "7290001000113",
    "7290001000120",
]

# 1 NEW barcode for retry_less
RETRY_LESS_BARCODES = [
    "7290001000106",
]

# 5 labels: 3 new + 2 from photo1 (neighbors)
RETRY_MORE_BARCODES = [
    "7290001000106",  # new
    "7290001000113",  # new
    "7290001000120",  # new
    "7290001000014",  # known neighbor from photo1
    "7290001000021",  # known neighbor from photo1
]

# 6 barcodes for aggregation photo1
AGG_PHOTO1_BARCODES = [
    "7290002000013",
    "7290002000020",
    "7290002000037",
    "7290002000044",
    "7290002000051",
    "7290002000068",
]

# 5 NEW barcodes for aggregation photo2
AGG_PHOTO2_BARCODES = [
    "7290002000075",
    "7290002000082",
    "7290002000099",
    "7290002000105",
    "7290002000112",
]


def render_code128(code: str) -> Image.Image:
    """Render a Code128 barcode encoding an EAN-13 value to a high-quality
    PIL image (black on white).

    Uses large module width and height for crisp, scannable barcodes.
    Returns an RGB image.
    """
    bc = CODE128(code, writer=ImageWriter())
    import io
    buf = io.BytesIO()
    bc.write(
        buf,
        options={
            "module_width": 0.5,   # wide bars for crisp edges
            "module_height": 50,   # tall bars
            "font_size": 24,
            "text_distance": 10,
            "quiet_zone": 10,
        },
    )
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_label(code: str, blur: bool = False) -> Image.Image:
    """Create a single box label: white background + barcode + text.

    The label is large enough for the scanner to read at high quality.
    """
    bc_img = render_code128(code)
    # Add padding around the barcode to simulate a real label
    label_w = bc_img.width + 80
    label_h = bc_img.height + 80
    label = Image.new("RGB", (label_w, label_h), "white")
    label.paste(bc_img, (40, 30))
    if blur:
        # Blur the barcode area heavily to make it unscannable
        bc_box = (40, 30, 40 + bc_img.width, 30 + bc_img.height)
        blurred = label.crop(bc_box).filter(ImageFilter.GaussianBlur(radius=12))
        label.paste(blurred, bc_box)
    return label


def compose_grid(labels: list[Image.Image], cols: int = 4) -> Image.Image:
    """Arrange labels in a grid with spacing, high quality.

    The final image is large (1500x2000+) to match real photo quality.
    """
    rows = (len(labels) + cols - 1) // cols
    label_w = max(lbl.width for lbl in labels)
    label_h = max(lbl.height for lbl in labels)
    padding = 60
    grid_w = cols * label_w + (cols + 1) * padding
    grid_h = rows * label_h + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), (200, 200, 200))  # gray bg
    for i, label in enumerate(labels):
        r = i // cols
        c = i % cols
        x = padding + c * (label_w + padding)
        y = padding + r * (label_h + padding)
        grid.paste(label, (x, y))
    return grid


def build_image(name: str, barcodes: list[str], blur_indices: set[int] | None = None):
    """Build a high-quality test image with labels for the given barcodes."""
    blur_indices = blur_indices or set()
    labels = [make_label(bc, blur=i in blur_indices) for i, bc in enumerate(barcodes)]
    grid = compose_grid(labels)
    out = OUT / name
    grid.save(out, "PNG", optimize=False)  # no compression artifacts
    print(f"  {out}: {len(barcodes)} labels ({len(blur_indices)} blurred) {grid.size}")


def main():
    print("Generating synthetic test images:")
    # Photo 1: 12 labels, 9 scannable, 3 blurred (indices 9,10,11)
    build_image("photo1_12_9found.png", PHOTO1_BARCODES, blur_indices={9, 10, 11})
    # Retry exact: 3 labels with the 3 missing barcodes
    build_image("retry_exact_3.png", RETRY_EXACT_BARCODES)
    # Retry less: 1 label with 1 missing barcode
    build_image("retry_less_1.png", RETRY_LESS_BARCODES)
    # Retry more: 5 labels (3 new + 2 known)
    build_image("retry_more_5.png", RETRY_MORE_BARCODES)
    # Aggregation photo1: 6 labels, all scannable
    build_image("agg_photo1_6.png", AGG_PHOTO1_BARCODES)
    # Aggregation photo2: 5 labels, all new
    build_image("agg_photo2_5.png", AGG_PHOTO2_BARCODES)
    print("Done.")


if __name__ == "__main__":
    main()
