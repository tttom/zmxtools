from collections.abc import Sequence
from pathlib import Path

from tests import log

log = log.getChild(__name__)

test_directory = Path(__file__).resolve().parent.parent / "data"

test_agf_files: Sequence[Path] = [_ for _ in (test_directory / "glass").glob("*") if _.suffix.lower() == ".agf"]
