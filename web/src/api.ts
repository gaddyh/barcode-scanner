export interface DetectedBarcode {
  value: string;
  format: string;
  content_type: string;
  orientation: number;
  position: { x: number; y: number }[];
  bounding_box: { x1: number; y1: number; x2: number; y2: number };
}

export interface ScanResponse {
  status: "found" | "not_found";
  count: number;
  image_width: number;
  image_height: number;
  filename: string;
  upload_bytes: number;
  elapsed_ms: number;
  barcodes: DetectedBarcode[];
}

// When VITE_API_BASE_URL is set (local dev with separate Vite server, or
// ngrok), use it. Otherwise use same-origin (empty string) — this is the
// Docker/Render case where the frontend and API are served from one origin.
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

export async function scanBarcode(file: File): Promise<ScanResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${apiBaseUrl}/barcode/scan`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail: string;
    try {
      const body = await res.json();
      detail = JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }

  return (await res.json()) as ScanResponse;
}
