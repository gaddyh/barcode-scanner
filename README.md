# Barcode Scanner Service

Deterministic barcode-scanning service for product photos. It is intentionally independent of WhatsApp and Priority ERP.

## Flow

```text
product photo -> FastAPI upload -> Pillow EXIF orientation normalization
  -> tiled multi-scale scan -> ZXing-C++ -> position-aware deduplication -> JSON
```

The scanner runs many decode passes per image to maximize recall on difficult
warehouse photos:

1. **Full-image pass** — scan the original image once.
2. **Tiling** — split the image into a 3×2 grid of overlapping tiles (20% overlap)
   and scan each tile independently so small barcodes fill more of the frame.
3. **Preprocessing variants** — for every region (full image + each tile), four
   variants are decoded:
   - Original (or upscaled 2× / 3× via LANCZOS)
   - Grayscale
   - Grayscale + 1.8× contrast
   - Grayscale + contrast + UnsharpMask sharpening
4. **Deduplication** — the same physical barcode is detected many times across
   tiles, scales, and variants. Detections are grouped by `(value, format)` and
   merged when their positions overlap (center-distance or IoU). The detection
   with the largest bounding box is kept. Results are sorted top-to-bottom,
   left-to-right.

ZXing-C++ handles rotation and inverted-image attempts internally via
`try_rotate` / `try_invert`.

## Requirements

- Python 3.11+
- No database
- No cloud vision account

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

## Scan a photo

```bash
curl -X POST http://localhost:8000/barcode/scan \
  -H "accept: application/json" \
  -F "file=@./product.jpg"
```

Example response:

```json
{
  "status": "found",
  "count": 2,
  "image_width": 2000,
  "image_height": 1500,
  "barcodes": [
    {
      "value": "7297501098442",
      "format": "Code 128",
      "content_type": "ContentType.Text",
      "orientation": -90,
      "position": [
        {"x": 619, "y": 878},
        {"x": 619, "y": 424},
        {"x": 559, "y": 415},
        {"x": 559, "y": 876}
      ],
      "bounding_box": {
        "x1": 559,
        "y1": 415,
        "x2": 619,
        "y2": 878
      }
    },
    {
      "value": "900439-42",
      "format": "Code 128",
      "content_type": "ContentType.Text",
      "orientation": 0,
      "position": [
        {"x": 1171, "y": 679},
        {"x": 1465, "y": 679},
        {"x": 1459, "y": 779},
        {"x": 1170, "y": 779}
      ],
      "bounding_box": {
        "x1": 1170,
        "y1": 679,
        "x2": 1465,
        "y2": 779
      }
    }
  ]
}
```

No barcode is still a successful API request:

```json
{
  "status": "not_found",
  "count": 0,
  "image_width": 2000,
  "image_height": 1500,
  "barcodes": []
}
```

## Scan from the command line

The `barcode-scan` console script (installed by `pip install -e ".[dev]"`) scans
one or more image files directly, with no HTTP server:

```bash
barcode-scan ./product.jpg
barcode-scan --pretty ./a.png ./b.png
```

Each image produces one JSON object in a top-level array. Unreadable files and
invalid images are reported per-entry with `"status": "error"` and the process
exits with a non-zero code if any image failed.

```json
[
  {
    "path": "./product.jpg",
    "status": "found",
    "count": 2,
    "barcodes": [
      {
        "value": "7297501098442",
        "format": "Code 128",
        "content_type": "ContentType.Text",
        "orientation": -90,
        "position": [
          {"x": 619, "y": 878},
          {"x": 619, "y": 424},
          {"x": 559, "y": 415},
          {"x": 559, "y": 876}
        ],
        "bounding_box": {
          "x1": 559,
          "y1": 415,
          "x2": 619,
          "y2": 878
        }
      }
    ]
  }
]
```

You can also run it without installing the entry point:

```bash
python -m app.cli ./product.jpg
```

## Run tests

```bash
pytest
```

The scanner tests monkeypatch `zxingcpp.read_barcodes` to inject deterministic
results, so they do not depend on a real barcode image. The CLI regression tests
run the actual `barcode-scan` command against the sample images in `samples/`.

## Docker

```bash
docker compose up --build
```

## Project layout

```text
app/
  api/routes.py                 HTTP validation and endpoint
  cli.py                        Command-line scanner (no HTTP server)
  core/config.py                Environment configuration
  models/barcode.py             API response contract (ScanResponse)
  services/barcode_scanner.py   BarcodeScanner — tiling, preprocessing, dedup
  main.py                       FastAPI application
samples/
  marny_brown_42.jpeg           Sample photo with 2 barcodes
  multi_clear_6_boxes.jpeg      Sample photo with 6 barcodes
tests/
  test_api.py                   API endpoint tests
  test_barcode_scanner.py       Scanner logic tests (monkeypatched zxing)
  test_cli.py                   CLI unit tests + regression tests on samples
```

## Current boundary

This version does not:

- call WhatsApp
- call Priority ERP
- use an LLM
- run object detection to find box regions (tiling is a fixed grid, not adaptive)
- validate barcode check digits
- expose per-detection confidence scores

The next step is to test it against real warehouse photos and record exact-match
success and failure reasons.
