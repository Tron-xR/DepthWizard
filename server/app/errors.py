"""Domain errors with stable error codes for the API error format."""
from __future__ import annotations


class DepthWizardError(Exception):
    """Base error. Subclasses define `error_code` used in API responses."""

    error_code = "error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
