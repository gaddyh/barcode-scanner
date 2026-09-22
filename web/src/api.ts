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

// --- Session types (/barcode/session) ---

export interface SessionItem {
  barcode_value: string;
  barcode_format: string | null;
  barcode_bbox: Record<string, number> | null;
  label_bbox: Record<string, number> | null;
  label_index: number | null;
  match_basis: string | null;
  source_image: number;
}

export interface SessionMissingItem {
  label_index: number | null;
  label_bbox: Record<string, number> | null;
  barcode_bbox: Record<string, number> | null;
  status: string;
  source_image: number;
  resolved: boolean;
}

export interface SelectOption {
  id: string;
  name: string;
}

export type OrderAction = "create_order" | "verify_order_before_shipment";

export interface SessionResult {
  session_id: string;
  status: "active" | "complete" | "expired" | "closed" | "failed" | "needs_user_selection";
  expected_count: number;
  found_count: number;
  missing_count: number;
  items: SessionItem[];
  missing: SessionMissingItem[];
  image_count: number;
  message: string | null;
  candidates: SessionItem[];
  customer_id: string | null;
  branch_id: string | null;
  action: OrderAction | null;
  latest_image: {
    image_index: number;
    status: string;
    found: SessionItem[];
    missing: SessionMissingItem[];
    visible_label_count: number;
    found_count: number;
    missing_count: number;
  } | null;
}

// --- Participant ID (one fresh session per page load) ---

// A fresh participant ID is generated on every page load, so each refresh
// starts a new session. The ID is stable across calls within the same page
// lifetime so create + upload + submit share the same session.
const _participantId = crypto.randomUUID();

export function getParticipantId(): string {
  return _participantId;
}

// Same-origin by default (Docker/Render). Set VITE_API_BASE_URL for local dev.
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

// ngrok free tier shows a browser warning page that blocks large POST requests.
// This header skips it. Harmless on other hosts.
const NGROK_SKIP_HEADER = { "ngrok-skip-browser-warning": "true" };

async function postFile(path: string, file: File): Promise<Response> {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${apiBaseUrl}${path}`, {
    method: "POST",
    body: form,
    headers: NGROK_SKIP_HEADER,
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

export async function fetchCustomers(): Promise<SelectOption[]> {
  const res = await fetch(`${apiBaseUrl}/customers`, { headers: NGROK_SKIP_HEADER });
  if (!res.ok) throw new Error(await extractError(res));
  const body = (await res.json()) as { items: SelectOption[] };
  return body.items;
}

export async function fetchBranches(customerId: string): Promise<SelectOption[]> {
  const res = await fetch(
    `${apiBaseUrl}/customers/${encodeURIComponent(customerId)}/branches`,
    { headers: NGROK_SKIP_HEADER },
  );
  if (!res.ok) throw new Error(await extractError(res));
  const body = (await res.json()) as { items: SelectOption[] };
  return body.items;
}

export async function submitSessionImage(
  file: File,
  customerId: string,
  branchId: string,
  action: OrderAction,
): Promise<SessionResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("customer_id", customerId);
  form.append("branch_id", branchId);
  form.append("action", action);
  form.append("participant_id", getParticipantId());
  const res = fetch(`${apiBaseUrl}/barcode/session`, {
    method: "POST",
    body: form,
  });
  const response = await res;
  if (!response.ok) {
    throw new Error(await extractError(response));
  }
  return (await response.json()) as SessionResult;
}

export async function getSession(sessionId: string): Promise<SessionResult> {
  const res = await fetch(`${apiBaseUrl}/barcode/session/${sessionId}`);
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as SessionResult;
}

export async function closeSession(sessionId: string): Promise<SessionResult> {
  const res = await fetch(`${apiBaseUrl}/barcode/session/${sessionId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as SessionResult;
}

export async function selectCandidate(
  barcodeValue: string,
): Promise<SessionResult> {
  const form = new FormData();
  form.append("participant_id", getParticipantId());
  form.append("barcode_value", barcodeValue);
  const res = await fetch(`${apiBaseUrl}/barcode/session/select`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as SessionResult;
}

// --- Receiving flow (/receiving) ---

export interface ReceivingSessionResponse {
  session_id: string;
  status: "active" | "submitting" | "submitted" | "submission_unknown";
  customer_id: string | null;
  branch_id: string | null;
  action: OrderAction | null;
  participant_id: string | null;
  box_count: number;
  expected_count: number;
  external_order_id: number | null;
  frozen: boolean;
  items: { barcode_value: string; barcode_format: string; quantity: number }[];
  discrepancy: {
    expected: number;
    found: number;
    missing: number;
    is_complete: boolean;
  };
}

export interface ReceivingImageResponse {
  session_id: string;
  status: string;
  outcome: string;
  boxes_added: number;
  total_boxes: number;
  expected_count: number;
  missing_count: number;
  discrepancy: {
    expected: number;
    found: number;
    missing: number;
    is_complete: boolean;
  };
  candidates: SessionItem[];
  message: string | null;
  annotated_image_b64?: string;
  annotated_image_width?: number;
  annotated_image_height?: number;
}

export interface ReceivingSubmitResponse {
  session_id: string;
  status: "active" | "submitting" | "submitted" | "submission_unknown";
  order_id: number | null;
  external_order_id: number | null;
  idempotent: boolean;
  items: { barcode_value: string; barcode_format: string; quantity: number }[];
  error: { code: string; message: string } | null;
  retry_recommended: boolean;
}

export async function createReceivingSession(
  participantId?: string,
  customerId?: string,
  branchId?: string,
  action?: OrderAction,
): Promise<ReceivingSessionResponse> {
  const form = new FormData();
  if (participantId) form.append("participant_id", participantId);
  if (customerId) form.append("customer_id", customerId);
  if (branchId) form.append("branch_id", branchId);
  if (action) form.append("action", action);
  const res = await fetch(`${apiBaseUrl}/receiving/sessions`, {
    method: "POST",
    body: form,
    headers: NGROK_SKIP_HEADER,
  });
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as ReceivingSessionResponse;
}

export async function attachReceivingSessionContext(
  sessionId: string,
  customerId: string,
  branchId: string,
  action: OrderAction,
): Promise<ReceivingSessionResponse> {
  const form = new FormData();
  form.append("customer_id", customerId);
  form.append("branch_id", branchId);
  form.append("action", action);
  const res = await fetch(
    `${apiBaseUrl}/receiving/sessions/${encodeURIComponent(sessionId)}/context`,
    { method: "POST", body: form, headers: NGROK_SKIP_HEADER },
  );
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as ReceivingSessionResponse;
}

export async function uploadReceivingImage(
  sessionId: string,
  file: File,
): Promise<ReceivingImageResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(
    `${apiBaseUrl}/receiving/sessions/${encodeURIComponent(sessionId)}/images`,
    { method: "POST", body: form, headers: NGROK_SKIP_HEADER },
  );
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as ReceivingImageResponse;
}

export async function getReceivingSession(
  sessionId: string,
): Promise<ReceivingSessionResponse> {
  const res = await fetch(
    `${apiBaseUrl}/receiving/sessions/${encodeURIComponent(sessionId)}`,
    { headers: NGROK_SKIP_HEADER },
  );
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as ReceivingSessionResponse;
}

export async function submitReceivingSession(
  sessionId: string,
): Promise<ReceivingSubmitResponse> {
  const res = await fetch(
    `${apiBaseUrl}/receiving/sessions/${encodeURIComponent(sessionId)}/submit`,
    { method: "POST", headers: NGROK_SKIP_HEADER },
  );
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as ReceivingSubmitResponse;
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
    headers: { "Content-Type": "application/json", ...NGROK_SKIP_HEADER },
    body: JSON.stringify({ trace_id: traceId, correct, comment }),
  });
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as FeedbackResponse;
}

async function extractError(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const body = JSON.parse(text);
    return `HTTP ${res.status}: ${JSON.stringify(body)}`;
  } catch {
    return `HTTP ${res.status}: ${text}`;
  }
}

// --- Admin metrics (/admin/metrics) ---

import type { MetricsResponse, GroupedMetricsResponse } from "./admin/types";

export async function fetchMetrics(hours: number): Promise<MetricsResponse> {
  const res = await fetch(`${apiBaseUrl}/admin/metrics?hours=${hours}`);
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as MetricsResponse;
}

export async function fetchGroupedMetrics(
  hours: number,
  groupBy: string,
): Promise<GroupedMetricsResponse> {
  const res = await fetch(
    `${apiBaseUrl}/admin/metrics?hours=${hours}&group_by=${groupBy}`,
  );
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return (await res.json()) as GroupedMetricsResponse;
}
