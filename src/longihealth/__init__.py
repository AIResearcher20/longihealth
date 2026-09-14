"""
LongiHealth — Reproducible Longitudinal EHR Analytics Pipeline.
"""

__version__ = "1.0.0"
__author__ = "Sepideh Moafi"

from .config import load_config

__all__ = ["load_config", "__version__"]
