class DAMError(Exception):
    """Base exception for the DAM package."""
    pass

class GateError(DAMError):
    """Raised when an image fails a heuristic or ML gate."""
    def __init__(self, code: str, message: str, metrics: dict = None):
        super().__init__(message)
        self.code = code
        self.metrics = metrics or {}

class ModelError(DAMError):
    """Raised when there is an issue with model loading or inference."""
    pass

class ConfigError(DAMError):
    """Raised when configuration is missing or invalid."""
    pass