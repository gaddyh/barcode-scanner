"""Create or update the barcode scanner LangSmith dashboard.

Usage:
    python scripts/provision_langsmith_dashboard.py --dry-run
    python scripts/provision_langsmith_dashboard.py
    python scripts/provision_langsmith_dashboard.py --check

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

from app.services.langsmith_dashboard import (
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
        help="Validate configuration and show whether the section exists; do not write.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the section and all expected charts exist; do not write.",
    )
    args = parser.parse_args()

    try:
        config = DashboardConfig.from_env()
        api = LangSmithDashboardApi(config)
        result = provision_dashboard(
            api,
            dry_run=args.dry_run,
            check_only=args.check,
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
        print("Dashboard provisioned.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
