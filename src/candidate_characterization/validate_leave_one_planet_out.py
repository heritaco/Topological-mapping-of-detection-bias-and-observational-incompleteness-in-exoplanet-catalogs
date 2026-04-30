"""Compatibility entrypoint for leave-one-planet-out validation.

The package now exposes validation through :mod:`validation`. This file keeps
the original drop-in surface available without duplicating the implementation.
"""

from __future__ import annotations

from .validation import validate_property_models

__all__ = ["validate_property_models"]
