import logging

from zmxtools import log, console_log_handler

__all__ = ['log']

console_log_handler.level = -1

log = log.getChild(__name__)
log.level = logging.DEBUG

log.info("TEST")
