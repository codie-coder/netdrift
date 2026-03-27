import logging
import structlog


def get_logger(name: str):
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    return structlog.get_logger(name)
