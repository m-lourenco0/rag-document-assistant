import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "file_debug": {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "level": "DEBUG",
            "filename": "debug.log",
            "mode": "w",
        },
    },
    "loggers": {
        "httpx": {"level": "WARNING"},
        "unstructured_inference": {"level": "WARNING"},
        "chromadb": {"level": "WARNING"},
        "timm": {"level": "WARNING"},
        "pikepdf": {"level": "WARNING"},
        "uvicorn.error": {"level": "WARNING"},
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file_debug"],
    },
}


def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
    logging.info(
        "Logging configured successfully. Console level: INFO, File level: DEBUG"
    )
