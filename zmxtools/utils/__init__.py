from zmxtools import log
from zmxtools.utils.factorial_fraction import factorial_fraction

__all__ = ['log', 'const_c', 'factorial_fraction']


log = log.getChild(__name__)

const_c = 299_792_458  # avoid importing scipy just for this
