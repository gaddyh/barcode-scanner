// --- Scanner-only response (/barcode/scan) ---

export interface DetectedBarcode {
  value: string;
  format: string;
  content_type: string;
  orientation: number;
  position: { x: number; y: number }[];
  bounding_box: { x1: number; y1: number; x2: number; y2: number };
}

export interface ScanResponse {
  upload_id: string;
  trace_id: string;
  source: string;
  status: "found" | "not_found";
  count: number;
  image_width: number;
  image_height: number;
  filename: string;
  upload_bytes: number;
  elapsed_ms: number;
  barcodes: DetectedBarcode[];
}

// --- Full pipeline response (/barcode/analyze) ---

export interface FoundItem {
  label_index: number;
  barcode_value: string;
  barcode_format: string;
  barcode_bbox: { x1: number; y1: number; x2: number; y2: number };
  label_bbox: { x1: number; y1: number; x2: number; y2: number };
  match_basis: string;
}

export interface MissingItem {
  label_index: number;
  status: string;
  label_bbox: { x1: number; y1: number; x2: number; y2: number } | null;
  barcode_bbox: { x1: number; y1: number; x2: number; y2: number } | null;
}

export interface UnassignedItem {
  barcode_value: string;
  barcode_format: string;
  barcode_bbox: { x1: number; y1: number; x2: number; y2: number };
}

export interface AnalyzeSummary {
  visible_label_count: number;
  found_count: number;
  missing_count: number;
  unassigned_count: number;
  all_found: boolean;
}

export interface AnalyzeResponse {
  ok: boolean;
  outcome: "complete" | "needs_better_photo" | "retryable_error";
  audit_available: boolean;
  image_width: number;
  image_height: number;
  upload_id: string;
  trace_id: string;
  source: string;
  filename: string;
  upload_bytes: number;
  elapsed_ms: number;
  found: FoundItem[];
  missing: MissingItem[];
  unassigned: UnassignedItem[];
  summary: AnalyzeSummary;
  error?: { code: string; message: string };
  annotated_image_b64?: string;
  annotated_image_width?: number;
  annotated_image_height?: number;
  message?: string;
}

// Same-origin by default (Docker/Render). Set VITE_API_BASE_URL for local dev.
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

async function postFile(path: string, file: File): Promise<Response> {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${apiBaseUrl}${path}`, {
    method: "POST",
    body: form,
  });
}

export async function scanBarcode(file: File): Promise<ScanResponse> {
  const res = await postFile("/barcode/scan", file);
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as ScanResponse;
}

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const res = await postFile("/barcode/analyze", file);
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as AnalyzeResponse;
}

// --- Feedback (/feedback) ---

export interface FeedbackResponse {
  status: string;
  trace_id: string;
  score: number;
}

export async function submitFeedback(
  traceId: string,
  correct: boolean,
  comment: string | null = null,
): Promise<FeedbackResponse> {
  const res = await fetch(`${apiBaseUrl}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trace_id: traceId, correct, comment }),
  });
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as FeedbackResponse;
}

async function extractError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return `HTTP ${res.status}: ${JSON.stringify(body)}`;
  } catch {
    return `HTTP ${res.status}: ${await res.text()}`;
  }
}
