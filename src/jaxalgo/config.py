"""Project Configuration."""
import os
import sys
from logging.config import dictConfig
from typing import NoReturn

basedir = os.path.abspath(os.path.dirname(__file__))
srcdir = os.path.abspath(os.path.join(basedir, os.pardir))
rootdir = os.path.abspath(os.path.join(srcdir, os.pardir))


class Config:
    """Provide default Configuration."""

    # logging
    LOG_LEVEL = "INFO"
    LOG_LINE_FORMAT = "%(asctime)s %(levelname)-5s %(threadName)s: %(message)s"
    LOG_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

    @classmethod
    def configure_logger(cls, root_module_name: str) -> NoReturn:
        """Configure logging."""
        dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "stdout_formatter": {
                    "format": cls.LOG_LINE_FORMAT,
                    "datefmt": cls.LOG_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "stdout_handler": {
                    "level": cls.LOG_LEVEL,
                    "formatter": "stdout_formatter",
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["stdout_handler"],
                    "level": cls.LOG_LEVEL,
                    "propagate": True,
                },
            },
        })

    # data
    DATADIR = os.path.join(rootdir, "data")


class TestConfig(Config):
    # pylint: disable=too-few-public-methods
    """Provide Testing Configuration."""
    LOG_LEVEL = "DEBUG"
