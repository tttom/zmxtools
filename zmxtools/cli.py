import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Sequence

from zmxtools import __version__, zar

log = logging.getLogger(__name__)

__all__ = ['unzar']

ZAR = '.zar'
ZIP = '.zip'


class CommandLineArgumentError(SyntaxError):
    """A class representing an error due to incorrect command line arguments."""


class UnzarArgumentParser(argparse.ArgumentParser):
    def __init__(self, name: str = 'unzar'):
        """Construct an argument parser specific for unzar."""
        super().__init__(
            description='Zemax archive files (.zar) unpacker and zipper.',
            usage=f"""
        > {name} filename.zar
        > {name} filename.zar -z
        > {name} -i filename.zar -o filename.zip
        > {name} -i filename.zar -o subfolder/filename.zip
        > {name} -i filename.zar -vv
        > {name} -h
        """,
            epilog='Error codes: 0 (OK), 1 (file skipped), 2 (incorrect argument), -1 (FATAL).\n' +
                   'Online: https://github.com/tttom/zmxtools, https://zmxtools.readthedocs.io',
            conflict_handler='resolve',
        )

        self.add_argument(
            '-v',
            '--verbosity',
            help='the level of verbosity',
            default=0,
            action='count',
        )
        self.add_argument(
            '-q',
            '--quiet',
            help='suppress output partially or completely (-qqqq)',
            default=0,
            action='count',
        )
        self.add_argument('-V', '--version', help='show version information before proceeding', action='store_true')
        self.add_argument('-i', '--input', type=str, nargs='*', help='the input archive')
        self.add_argument('-o', '--output', type=str, help='destination directory for the extracted archive')
        self.add_argument('-f', '--force', help='overwrite existing files if necessary', action='store_true')
        self.add_argument('-z', '--zip', help='create an archive instead of a directory', action='store_true')
        self.add_argument('archive_file_name', type=str, nargs='*', help='one or more names of input archives')

    def exit(self, status: int = 0, message: Optional[str] = None):
        """
        Custom handling of parser exit.

        :param status: The error code given by the parent class argparse.ArgumentParser.
        :param message: The message given by the parent class argparse.ArgumentParser.
        """
        raise CommandLineArgumentError(f'Error {status}, unrecognized command line argument: {message}')


def unzar(argv: Optional[Sequence[str]] = None) -> int:
    """
    Function that can be called as a script.

    :param argv: An optional sequence of input arguments.
    :returns: The error code. 0: no error, 1: file skipped, 2: incorrect argument, -1: unexpected fatal error.
    """
    # Parse the input arguments
    try:
        input_parser: UnzarArgumentParser = UnzarArgumentParser()
        input_args: argparse.Namespace = input_parser.parse_args(argv)

        verbosity_level: int = 2 + input_args.verbosity - input_args.quiet  # between -inf and +inf, default 2==WARNING
        log_levels: List[int] = [logging.FATAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
        log.level = log_levels[max(0, min(len(log_levels) - 1, verbosity_level))]

        log.debug(f'Parsed input arguments: {input_args}')

        version_message: str = (f'Running zmxtools unzar version {__version__}. ' +
                                'More info at: https://github.com/tttom/zmxtools'
                                )
        log.debug(version_message)
        if input_args.version and verbosity_level >= 0:
            sys.stdout.write(version_message)  # i.e. a print that lints

        combined_input_files: List[str] = input_args.input if input_args.input is not None else []
        combined_input_files += input_args.archive_file_name if input_args.archive_file_name is not None else []
        if len(combined_input_files) == 0:
            log.error('No input zar archives specified.')
            if verbosity_level >= 0:
                input_parser.print_usage()
            return 2
        log.info(f'Unpacking {len(combined_input_files)} archive(s)...')
        log.debug(f'with file names {combined_input_files}')
        nb_files_skipped = 0
        for input_file in combined_input_files:
            log.info(f'Processing {input_file}...')
            input_full_file = Path(input_file.strip())
            if input_full_file.suffix.lower() != ZAR:
                log.warning(f'Archive file "{input_full_file}" does not have the ".zar" extension!')
            # Check for direction on where to send the output
            if input_args.output is None:
                output_full_file = input_full_file.parent / (
                    input_full_file.stem if input_full_file.suffix.lower() == ZAR else input_full_file
                )
            else:
                output_full_file = Path(input_args.output.strip()) / input_full_file.stem
                log.debug(f'Writing output to specified output directory: {output_full_file.parent}...')
            # Delegate the actual work
            if input_args.zip:
                output_full_file = output_full_file.with_suffix(ZIP)
                if input_args.force and output_full_file.exists():
                    log.info(f'The file "{output_full_file}" already exists, but forcing an overwrite...')
                    output_full_file.unlink()
                if output_full_file.exists():
                    log.warning(f'The file "{output_full_file}" already exists, skipping it.' +
                                ' Use unzar --force to overwrite.',
                                )
                    nb_files_skipped += 1
                else:
                    zar.repack(input_full_file, output_full_file)
            else:
                zar.extract(input_full_file, output_full_file)
    except CommandLineArgumentError as clae:
        log.fatal(clae)
        return 2
    except Exception as exc:
        log.fatal(f'An unexpected fatal error occured: "{exc}"')
        log.info(traceback.format_exc())
        return -1

    # No error unless at least one file skipped
    return min(1, nb_files_skipped)


if __name__ == '__main__':
    sys.exit(unzar())
