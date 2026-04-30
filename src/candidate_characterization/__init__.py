"""Candidate characterization package for topological exoplanet incompleteness projects.

This package extends the existing Mapper/TDA pipeline with probabilistic property
characterization for candidate missing planets. It is intentionally conservative:
it estimates distributions and support diagnostics, not confirmed planet detections.
"""

from .version import __version__

__all__ = ["__version__"]
