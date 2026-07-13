from __future__ import annotations

import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "application": {
            "()": logging.Formatter,
            "fmt": (
                "%(asctime)s %(levelname)s %(name)s %(message)s "
                "request_id=%(request_id)s operation=%(operation)s "
                "duration_ms=%(duration_ms)s"
            ),
            "defaults": {
                "request_id": "-",
                "operation": "-",
                "duration_ms": "-",
            },
        }
    },
    "handlers": {
        "application": {
            "class": "logging.StreamHandler",
            "formatter": "application",
        }
    },
    "loggers": {
        "api": {
            "handlers": ["application"],
            "level": "INFO",
            "propagate": False,
        },
        "doc_assistant": {
            "handlers": ["application"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def configure_logging() -> None:
    dictConfig(LOGGING_CONFIG)
