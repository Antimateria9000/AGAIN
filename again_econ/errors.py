class AgainEconError(Exception):
    """Base exception for the economic backtesting module."""


class BacktestConfigurationError(AgainEconError):
    """Raised when the runtime configuration is invalid."""


class ContractValidationError(AgainEconError):
    """Raised when an input contract is malformed."""


class TemporalIntegrityError(AgainEconError):
    """Raised when an input violates temporal integrity guarantees."""


class AdapterError(AgainEconError):
    """Raised when a forecast or signal bundle cannot be loaded."""


class ExecutionError(AgainEconError):
    """Raised when an execution step cannot be simulated coherently."""
