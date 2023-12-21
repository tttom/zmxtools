import sys
import logging
from importlib.metadata import PackageNotFoundError, version
import coloredlogs


__all__ = ["__version__", "zar", "log", "console_log_handler"]

coloredlogs.enable_ansi_support()

__field_styles = coloredlogs.DEFAULT_FIELD_STYLES
__field_styles["msecs"] = __field_styles["asctime"]
__field_styles["levelname"] = dict(color='green')
__level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.update(
    spam=dict(color="blue", faint=True),
    debug=dict(color="blue"),
    verbose=dict(color="blue", bold=True),
    info=dict(),
    warning=dict(color=(255, 64, 0)),
    error=dict(color=(255, 0, 0)),
    fatal=dict(color=(255, 0, 0), bold=True, background=(255, 255, 0)),
    critical=dict(color=(0, 0, 0), bold=True, background=(255, 255, 0))
)

__formatter = coloredlogs.ColoredFormatter(f'%(asctime)s|%(name)s-%(levelname)s: %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S.%f',
                                           field_styles=__field_styles, level_styles=__level_styles)

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

log = logging.getLogger(__name__)  # create logger
log.level = -1

# create console handler
console_log_handler = logging.StreamHandler(sys.stdout)
# create formatter and add it to the handlers
console_log_handler.formatter = __formatter
console_log_handler.level = logging.WARNING
log.addHandler(console_log_handler)  # add the handler to the logger
