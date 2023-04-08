import logging
from importlib.metadata import PackageNotFoundError, version

try:
    import coloredlogs
    formatter_class = coloredlogs.ColoredFormatter
except ImportError:
    formatter_class = logging.Formatter

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = ['__version__', 'zar']

# create logger
log = logging.getLogger(__name__)
log.level = logging.WARNING

# create formatter and add it to the handlers
_formatter = formatter_class('%(asctime)s|%(name)s-%(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')
_debug_formatter = logging.Formatter(  # Don't use colored logs
    '%(asctime)s|%(name)s-%(levelname)5s %(threadName)s:%(filename)s:%(lineno)s:%(funcName)s| %(message)s',
)

# create console handler
_ch = logging.StreamHandler()
_ch.setFormatter(_formatter)
log.addHandler(_ch)  # add the handler to the logger
