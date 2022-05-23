import coloredlogs
import logging


formatter_class = coloredlogs.ColoredFormatter

# create logger
log = logging.getLogger(__name__)
log.level = logging.INFO

# create formatter and add it to the handlers
_formatter = formatter_class('%(asctime)s|%(name)s-%(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')
_debug_formatter = logging.Formatter(  # Don't use colored logs
    '%(asctime)s|%(name)s-%(levelname)5s %(threadName)s:%(filename)s:%(lineno)s:%(funcName)s| %(message)s',
)

# create console handler
_ch = logging.StreamHandler()
_ch.setFormatter(_formatter)
log.addHandler(_ch)  # add the handler to the logger
