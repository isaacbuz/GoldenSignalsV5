# Re-export settings from config.py for backward compatibility
from .config import settings, validate_settings

__all__ = ["settings", "validate_settings"]