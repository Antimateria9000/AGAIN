class AgainBenchmarkError(Exception):
    """Base exception for the reproducible benchmark module."""


class BenchmarkValidationError(AgainBenchmarkError):
    """Raised when a benchmark definition, snapshot or manifest is invalid."""


class BenchmarkStorageError(AgainBenchmarkError):
    """Raised when benchmark artifacts cannot be persisted or loaded."""


class BenchmarkExecutionError(AgainBenchmarkError):
    """Raised when a benchmark run cannot be executed coherently."""


class BenchmarkAdapterError(AgainBenchmarkError):
    """Raised when the AGAIN adapter cannot provide data or predictions."""
