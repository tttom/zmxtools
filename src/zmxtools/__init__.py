"""A toolkit to read Zemax files.

Public attributes:

    :mod:`zar`: A module to load zar files and store their contents in a standard format.

    :mod:`zmx`: A module to parse zmx files as a Python object of :class:`OpticalDesign`.

    :mod:`agf`: A module to parse zmx files as a Python object of :class:`MaterialLibrary`.

    :func:`log_to_file`: Call this function at the start of your script to save all log messages to a file.

    :attr:`log`: The :mod:`logging` object of this library. This can be used to log time-stamped messages from your code, or make the library's messages more or less verbose.

    :attr:`console_log_handler`: The default :class:`logging.StreamHandler` that is used for console log messages. This can be used to change the verbosity of the output to the terminal.

    :data:`__version__`: The library's version number as a `str`. This is useful for debugging.

"""
import importlib.metadata
import logging
import logging.handlers
import pathlib
import sys

import coloredlogs

__all__ = ['__version__', 'agf', 'config_logger', 'log', 'log_to_file', 'zar', 'zmx']

__version__ = importlib.metadata.version(__name__)


# Create the console log handler
console_log_handler = logging.StreamHandler(sys.stdout)
# Setup the color log formatter
coloredlogs.enable_ansi_support()
field_styles = dict(coloredlogs.DEFAULT_FIELD_STYLES)
field_styles['msecs'] = field_styles['asctime']
field_styles['levelname'] = {'color': 'green'}
level_styles = dict(
    coloredlogs.DEFAULT_LEVEL_STYLES,
    spam={'color': 'blue', 'faint': True},
    debug={'color': 'blue', 'bright': True},
    verbose={'color': 'blue', 'bold': True},
    warning={'color': 'yellow'},
    error={'color': 'red', 'bright': True, 'bold': True},
    critical={'color': 'red', 'bold': True, 'background': 'yellow'}
)
color_formatter = coloredlogs.ColoredFormatter(
    '%(asctime)s|%(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S.%f',
    field_styles=field_styles, level_styles=level_styles
)
console_log_handler.setFormatter(color_formatter)

logging.captureWarnings(True)

def config_logger(
    name: str | None = None,
    level: int | str = logging.DEBUG
    ) -> logging.Logger:
    """Configures the logger with the specified name and outputs messages to the console.

    Note that this configures the root logger to log to the console. Calling `logging.basicConfig` will not make any further changes.

    Args:
        name:
            The name of the :class:`Logger` to return. Default: The currently executing file without the extension.
        level:
            The verbosity of on-screen messages in the terminal. Default: `DEBUG`, everything is displayed.
            Set to `INFO`, `WARNING`, `ERROR`, or `CRITICAL` to see fewer messages.

    Returns:
        A :class:`logging.Logger` object with the specified name.
    """
    if name is None:
        name = pathlib.Path(sys.argv[0]).stem

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Configure the root logger
    root_logger = logging.getLogger()
    console_log_handler.setLevel(level)
    if console_log_handler not in root_logger.handlers:  # Avoid duplicate output when called more than once.
        root_logger.addHandler(console_log_handler)

    return logger

# Create the library logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Map of log file path -> file handler so that log_to_file() never adds a duplicate handler for the same file.
_file_log_handlers: dict[pathlib.Path, logging.Handler] = {}

def log_to_file(
    file_path: pathlib.Path | str | None = None,
    name: str | None = None,
    level: int | str = logging.DEBUG,
    console_level: int | str = logging.DEBUG
    ) -> logging.Logger:
    """Write all log messages to the specified file.

    Args:
        file_path:
            The file to write the logs to. The extension is always changed to '.log', 
            so you can use `__file__` as an argument to create a log file named as your script.
            If the file exists, new messages are appended. Default: The currently executing file with the '.log' extension.
        name:
            The name of the :class:`Logger` to return. Default: the filename.
        level:
            The verbosity of the logs in the file. Default: `DEBUG`, everything is written to file.
            Set to `INFO`, `WARNING`, `ERROR`, or `CRITICAL` to see fewer messages.
        console_level:
            The verbosity of on-screen messages in the terminal. Default: `DEBUG`, everything is displayed.
            Set to `INFO`, `WARNING`, `ERROR`, or `CRITICAL` to see fewer messages.

    Returns:
        A :class:`logging.Logger` object with the specified name.
    """
    if file_path is None:
        file_path = sys.argv[0]
    file_path = pathlib.Path(file_path).with_suffix('.log').resolve()

    # Create a file handler, or reuse the existing one if this file is already being logged to, to avoid duplicate output.
    fh = _file_log_handlers.get(file_path)
    if fh is None:
        monochrome_formatter = logging.Formatter('%(asctime)s.%(msecs).03d|%(name)s-%(levelname)5s %(threadName)s:%(filename)s:%(lineno)s:%(funcName)s| %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.handlers.RotatingFileHandler(file_path, encoding='utf-8', maxBytes=4 * 1024 ** 2, backupCount=4)
        fh.setFormatter(monochrome_formatter)
        _file_log_handlers[file_path] = fh

    # Set up root logger
    root_logger = logging.getLogger()
    fh.setLevel(level)
    if fh not in root_logger.handlers:
        root_logger.addHandler(fh)

    logger = config_logger(name=name, level=console_level)

    log.info(f"{'Append' if file_path.exists() else 'Writ'}ing log messages to {file_path}...")

    return logger

# Import these later to avoid circular import errors:
from . import agf, zar, zmx
