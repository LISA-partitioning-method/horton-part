# HORTON-PART: molecular density partition schemes based on HORTON package.
# Copyright (C) 2023-2025 The HORTON-PART Development Team
#
# This file is part of HORTON-PART
#
# HORTON-PART is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON-PART is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
import logging
import os
import sys

__all__ = ["deflist", "setup_logger", "get_print_func"]


def deflist(logger: logging.Logger, l: list) -> None:
    """Print a definition list.

    Parameters
    ----------
    logger: logging.Logger
        The logger.
    l : list
        A list of keyword and value pairs. A table will be printed where the first
        column contains the keywords and the second column contains (wrapped) values.
    """
    widest = max(len(item[0]) for item in l)
    for name, value in l:
        logger.info(f"  {name.ljust(widest)} : {value}")


def setup_logger(logger, log_level=logging.INFO, log_file=None, overwrite=True):
    """
    Set up a logger with specified log level and log file.

    This function configures the provided logger to output logs to either a specified file or,
    if no file is specified, to the standard output (console). It first removes any existing
    handlers to prevent duplicate logging and then adds a new handler according to the
    specified settings.

    Parameters
    ----------
    logger : logging.Logger
        The logger to be configured.
    log_level : int, optional
        The logging level (e.g., logging.DEBUG, logging.INFO, etc.), by default logging.INFO.
    log_file : str or None, optional
        Path to the file where logs should be written. If None, logs will be written to
        standard output (console), by default None.
    overwrite : bool, optional
        Whether overwrite log file if it already exists.
    """
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create a logger
    logger.setLevel(log_level)

    # Remove any existing handlers. This is improved to handle cases where handlers might be absent.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set up logging formatter
    formatter = logging.Formatter(
        "%(levelname)s: %(message)s" if log_level <= logging.DEBUG else "%(message)s"
    )

    # Configure file handler if fn_log is provided, else use stream handler
    if log_file:
        path = os.path.dirname(log_file)
        if path and not os.path.exists(path):
            os.makedirs(path)

        mode = "w"
        if os.path.exists(log_file) and not overwrite:
            mode = "a"
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def get_print_func(logger=None, verbose=False):
    if logger is None:
        print_func = print
    else:
        print_func = logger.info if verbose else logger.debug
    return print_func
