import logging
from logging.handlers import TimedRotatingFileHandler
from enum import Enum
import sys

LOG_FILE_PATH = "data_drift_detector.log"
LOGGING_LEVEL = logging.DEBUG
FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class LoggerType(Enum):
    CONSOLE = 1
    FILE = 2


class LoggerHandler:
    def __init__(self, logger_handler_type: LoggerType, formatter: logging.Formatter,
                 log_level: logging) -> None:
        self._logger_handler_type = logger_handler_type
        self._formatter = formatter
        self._log_level = log_level
        self._handler = self._get_logger_handler()

    def _get_logger_handler(self) -> logging.Handler:
        handler = None

        if self._logger_handler_type == LoggerType.CONSOLE:
            handler = logging.StreamHandler(sys.stdout)

        elif self._logger_handler_type == LoggerType.FILE:
            handler = TimedRotatingFileHandler(LOG_FILE_PATH, when='midnight')

        assert handler is not None

        handler.setFormatter(self._formatter)
        handler.setLevel(level=self._log_level)

        return handler

    @property
    def handler(self):
        return self._handler

    @handler.setter
    def handler(self, value):
        self._handler = value


class Logger:

    def __init__(self, logger_name: str):
        self._logger = logging.getLogger(logger_name)

        self._logger.addHandler(LoggerHandler(LoggerType.CONSOLE,
                                              FORMATTER,
                                              LOGGING_LEVEL).handler)

        self._logger.addHandler(LoggerHandler(LoggerType.FILE,
                                              FORMATTER,
                                              LOGGING_LEVEL).handler)
        self._logger.setLevel(LOGGING_LEVEL)

        # with this pattern, it's rarely necessary to propagate the error up to parent
        self._logger.propagate = False

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value


def get_logger(logger_name):
    return Logger(logger_name).logger
