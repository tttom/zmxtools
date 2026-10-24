import argparse
import logging
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import zmxtools
from zmxtools import zar

log = zmxtools.config_logger(__name__)

__all__ = ["unzar"]

ZAR = ".zar"
ZIP = ".zip"


class CommandLineArgumentError(SyntaxError):
    """A class representing an error due to incorrect command line arguments."""


class UnzarArgumentParser(argparse.ArgumentParser):
    def __init__(self, name: str = "unzar"):
        """Construct an argument parser specific for unzar."""
        super().__init__(
            description="Zemax archive files (.zar) unpacker, converter, and inspector.",
            usage=f"""
> {name} filename.zar                               # Extract the contents of the ZAR file to a directory with the same name.
> {name} filename.zar -z                            # Convert the ZAR file to a standard zip file.
> {name} filename.zar -o filename.zip               # Explicitly specify the output files. Creating ZAR files is not supported.
> {name} -i filename.zar -o subfolder/filename.zip  # Explicitly specify the input and the output files. Full paths can be used.
> {name} filename.zar -f                            # Force overwrite any existing files.
> {name} filename.zar -v (or -vv)                   # Increase the verbosity of the console output messages.
> {name} filename.zar -qqq                          # Suppress warnings and errors on the console.
> {name} filename.zar -s                            # Show the optical model as an interactive 3D webview using oshow. This also works with zip files and directories.
""",
            epilog="Error codes: 0 (OK), 1 (file skipped), 2 (incorrect argument), -1 (FATAL).\n"
            + "Online source: https://github.com/tttom/zmxtools, and docs: https://zmxtools.readthedocs.io.",
            conflict_handler="resolve",
        )

        self.add_argument(
            "-v",
            "--verbosity",
            help="the level of verbosity",
            default=0,
            action="count",
        )
        self.add_argument(
            "-q",
            "--quiet",
            help="suppress output partially or completely (-qqqq)",
            default=0,
            action="count",
        )
        self.add_argument("-V", "--version", help="show version information before proceeding", action="store_true")
        self.add_argument("-i", "--input", type=str, nargs="*", help="the input archive")
        self.add_argument("-o", "--output", type=str, help="destination directory for the extracted archive")
        self.add_argument("-f", "--force", help="overwrite existing files if necessary", action="store_true")
        self.add_argument("-z", "--zip", help="create an archive instead of a directory", action="store_true")
        self.add_argument(
            "-s",
            "--show",
            help="display the unpacked module using oshow",
            action="store_true",
        )
        self.add_argument("archive_file_name", type=str, nargs="*", help="one or more names of input archives")

    def exit(self, status: int = 0, message: str | None = None):
        """
        Custom handling of parser exit.

        Args:
            status: The error code given by the parent class argparse.ArgumentParser.
            message: The message given by the parent class argparse.ArgumentParser.

        """
        if status != 0:
            raise CommandLineArgumentError(f"Error {status}, unrecognized command line argument: {message}")


def unzar(argv: Sequence[str] | None = None) -> int:
    """
    Function that can be called as a script.

    Args:
        argv: An optional sequence of input arguments.

    Returns:
        The error code. 0: no error, 1: file skipped, 2: incorrect argument, -1: unexpected fatal error.

    """
    # Parse the input arguments
    try:
        parser: UnzarArgumentParser = UnzarArgumentParser()
        args: argparse.Namespace = parser.parse_args(argv)

        verbosity_level: int = args.verbosity - args.quiet
        log_levels = logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG, logging.NOTSET
        DEFAULT_VERBOSITY: int = 2
        zmxtools.console_log_handler.level = log_levels[max(0, min(len(log_levels) - 1, DEFAULT_VERBOSITY + verbosity_level))]  # Default to WARNING

        log.debug(f"Parsed input arguments: {args}")

        version_message: str = f"Running zmxtools unzar version {zmxtools.__version__}."
        log.debug(version_message)
        if args.version:
            if verbosity_level >= -DEFAULT_VERBOSITY:
                sys.stdout.write(version_message + "\n")  # i.e. a print that lints
            if args.input is None and (args.archive_file_name is None or len(args.archive_file_name) == 0):
                return 0

        combined_input_files: list[str] = args.input if args.input is not None else []
        combined_input_files += args.archive_file_name if args.archive_file_name is not None else []
        if len(combined_input_files) == 0:
            log.error("No input zar archives specified.")
            if verbosity_level >= -DEFAULT_VERBOSITY:
                parser.print_usage()
            return 2
        log.info(f"Unpacking {len(combined_input_files)} archive(s)...")
        log.debug(f"with file names {combined_input_files}")
        nb_files_skipped = 0
        unpacked_outputs: list[Path] = []
        for input_file in combined_input_files:
            log.debug(f"Processing {input_file}...")
            input_full_file = Path(input_file.strip())
            if input_full_file.suffix.lower() != ZAR:
                log.warning(f'Archive file "{input_full_file}" does not have the ".zar" extension!')
            # Check for direction on where to send the output
            if args.output is None:
                output_full_file = input_full_file.parent / (
                    input_full_file.stem if input_full_file.suffix.lower() == ZAR else input_full_file
                )
            else:
                output_full_file = Path(args.output.strip()) / input_full_file.stem
                log.debug(f"Writing output to specified output directory: {output_full_file.parent}...")
            # Delegate the actual work
            if args.zip:
                output_full_file = output_full_file.with_suffix(ZIP)
                if args.force and output_full_file.exists():
                    log.info(f'The file "{output_full_file}" already exists, but forcing an overwrite...')
                    output_full_file.unlink()
                if output_full_file.exists():
                    log.warning(
                        f'The file "{output_full_file}" already exists, skipping it.'
                        + " Use unzar --force to overwrite.",
                    )
                    nb_files_skipped += 1
                else:
                    zar.repack(input_full_file, output_full_file)
                    unpacked_outputs.append(output_full_file)
            else:
                zar.extract(input_full_file, output_full_file)
                unpacked_outputs.append(output_full_file)
        if args.show:
            from zmxtools import gui

            for output in unpacked_outputs:
                log.info(f"Showing {output} in oshow...")
                show_code = gui.oshow([str(output)])
                if show_code != 0:
                    log.warning(f"oshow exited with code {show_code} for {output}.")
    except CommandLineArgumentError as clae:
        log.error(clae)
        return 2
    except Exception as exc:
        log.fatal(f'An unexpected fatal error occurred: "{exc!r}"')
        log.info(traceback.format_exc())
        return -1

    # No error unless at least one file skipped
    return min(1, nb_files_skipped)


if __name__ == "__main__":
    sys.exit(unzar())
