// DTO interfaces matching the /admin/metrics API response shapes.
// Kept separate from api.ts for clean separation of types vs fetch logic.

export interface VersionRate {
  version: string;
  total: number;
  rate_pct: number;
}

export interface MetricsResponse {
  time_window_hours: number;
  source: string;
  truncated: boolean;

  // Operations
  images_processed: number;
  boxes_processed: number;
  first_pass_complete_pct: number;
  final_complete_pct: number;
  recovery_attempted_pct: number;
  recovery_success_pct: number;
  user_retry_required_pct: number;
  p95_latency_ms: number;

  // Quality
  scanner_vision_match_pct: number;
  avg_count_delta: number;
  avg_missing_count: number;
  avg_unassigned_count: number;
  recovered_complete_pct: number;
  still_incomplete_pct: number;

  // Recovery detail
  avg_labels_tried: number;
  avg_labels_resolved: number;

  // Issues
  primary_issue_counts: Record<string, number>;

  // Version breakdowns
  completion_by_pipeline_version: VersionRate[];
  mismatch_by_scanner_version: VersionRate[];
  recovery_success_by_recovery_version: VersionRate[];
  retry_by_vision_model: VersionRate[];
}

export interface GroupedMetricsResponse {
  time_window_hours: number;
  source: string;
  truncated: boolean;
  group_by: string;
  groups: Record<string, MetricsResponse>;
}
