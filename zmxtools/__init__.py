import coloredlogs
import logging
import pkg_resources


formatter_class = coloredlogs.ColoredFormatter

__version__ = pkg_resources.get_distribution(__name__).version

__all__ = ['zar']

# create logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
_formatter = formatter_class('%(asctime)s|%(name)s-%(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')
_debug_formatter = logging.Formatter(  # Don't use colored logs
    '%(asctime)s|%(name)s-%(levelname)5s %(threadName)s:%(filename)s:%(lineno)s:%(funcName)s| %(message)s',
)

# Clear all previously added handlers
for _ in log.handlers:
    log.removeHandler(_)

# create console handler
_ch = logging.StreamHandler()
_ch.setFormatter(_formatter)
log.addHandler(_ch)  # add the handler to the logger
