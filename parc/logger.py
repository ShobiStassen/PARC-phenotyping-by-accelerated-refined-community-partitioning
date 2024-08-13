"""Custom colored logger.

See:

`StackOverflow <http://stackoverflow.com/a/24956305/408556>`__
`python-colors GitHub <https://gist.github.com/dideler/3814182>`__

"""

import logging
import sys

MIN_LEVEL = logging.DEBUG
MESSAGE = 25
logging.addLevelName(MESSAGE, "MESSAGE")
LOGGING_LEVEL = 20


class LogFilter(logging.Filter):
    """Filters (lets through) all messages with level < LEVEL"""
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        # "<" instead of "<=": since logger.setLevel is inclusive, this should
        # be exclusive
        return record.levelno < self.level


class Logger(logging.Logger):
    def message(self, msg, *args, **kwargs):
        if self.isEnabledFor(MESSAGE):
            self._log(MESSAGE, msg, args, **kwargs)


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt_prefix, fmt_msg):
        self.use_color = self.supports_color()
        self.fmt_prefix = fmt_prefix
        self.fmt_msg = fmt_msg
        super().__init__(fmt=f"{fmt_prefix} {fmt_msg}")

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.WARNING:
            if self.use_color:
                fmt = f"\x1b[93m{self.fmt_prefix}\x1b[0m {self.fmt_msg}"
                formatter = logging.Formatter(fmt)
                return formatter.format(record)
            else:
                return super().format(record)
        elif record.levelno == logging.ERROR:
            if self.use_color:
                fmt = f"\x1b[91m{self.fmt_prefix}\x1b[0m {self.fmt_msg}"
                formatter = logging.Formatter(fmt)
                return formatter.format(record)
            else:
                return super().format(record)
        elif record.levelno == 25 or record.levelno == logging.INFO:
            if self.use_color:
                fmt = f"\x1b[96m{self.fmt_prefix}\x1b[0m {self.fmt_msg}"
                formatter = logging.Formatter(fmt)
                return formatter.format(record)
            else:
                return super().format(record)

    def supports_color(self) -> bool:
        """Check if the system supports ANSI color formatting."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def get_logger(module_name, level: int = LOGGING_LEVEL) -> Logger:
    logging.setLoggerClass(Logger)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stdout_handler.addFilter(LogFilter(logging.WARNING))
    stdout_handler.setLevel(level)
    if level == 25:
        formatter = ColoredFormatter(
            fmt_prefix="[%(levelname)s]:",
            fmt_msg="%(message)s"
        )
    else:
        formatter = ColoredFormatter(
            fmt_prefix="[%(levelname)s] %(name)s.%(funcName)s (line %(lineno)d):",
            fmt_msg="%(message)s"
        )
    stdout_handler.setFormatter(formatter)
    stderr_handler.setLevel(max(MIN_LEVEL, logging.WARNING))
    # messages lower than WARNING go to stdout
    # messages >= WARNING (and >= STDOUT_LOG_LEVEL) go to stderr
    logger = logging.getLogger(module_name)
    logger.propagate = False
    logger.handlers = [stdout_handler, stderr_handler]
    logger.setLevel(level)
    return logger
