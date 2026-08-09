"""Create or update the barcode scanner LangSmith dashboards.

Four dashboards:
    1. Operations   — Is the system healthy?
    2. Quality      — Where are we losing boxes?
    3. Recovery     — Is recovery worth its complexity?
    4. Versions     — Did a new version actually improve things?

Usage:
    python scripts/provision_langsmith_dashboard.py --dry-run
    python scripts/provision_langsmith_dashboard.py
    python scripts/provision_langsmith_dashboard.py --check
    python scripts/provision_langsmith_dashboard.py --dashboard barcode-scanner-operations

Required environment variables:
    LANGSMITH_API_KEY
    LANGSMITH_PROJECT_ID  # tracing project UUID

Optional:
    LANGSMITH_ENDPOINT     # defaults to https://api.smith.langchain.com
    LANGSMITH_TENANT_ID
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from src.observability.dashboard import (
    DASHBOARDS,
    DashboardApiError,
    DashboardConfig,
    LangSmithDashboardApi,
    provision_dashboard,
)

load_dotenv()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show whether sections exist; do not write.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify all sections and charts exist; do not write.",
    )
    parser.add_argument(
        "--dashboard",
        type=str,
        default=None,
        help="Provision only one dashboard by key (e.g. barcode-scanner-operations). "
        "Default: provision all.",
        choices=[d.key for d in DASHBOARDS] + [None],
    )
    args = parser.parse_args()

    try:
        config = DashboardConfig.from_env()
        api = LangSmithDashboardApi(config)
        result = provision_dashboard(
            api,
            dry_run=args.dry_run,
            check_only=args.check,
            dashboard_key=args.dashboard,
        )
    except (ValueError, DashboardApiError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    if args.dry_run:
        print("Dry run: no dashboard resources were changed.", file=sys.stderr)
    elif args.check:
        print("Dashboard check passed.", file=sys.stderr)
    else:
        print("Dashboard(s) provisioned.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
