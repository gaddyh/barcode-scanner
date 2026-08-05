"""Re-export settings from the single source of truth in ``app.config``.

The barcode-scanner product API routes (``app/api/routes.py``) import
``Settings`` and ``get_settings`` from here via FastAPI ``Depends``.
All environment loading now lives in ``app/config.py`` so there is one
settings class and one env loader, not two.
"""

from app.config import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
