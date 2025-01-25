import logging

from zmxtools import console_log_handler, log

__all__ = ['log']

console_log_handler.level = -1

log = log.getChild(__name__)
log.level = logging.DEBUG

log.info('TEST')
