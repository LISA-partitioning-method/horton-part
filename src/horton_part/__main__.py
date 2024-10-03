# HORTON-PART: molecular density partition schemes based on HORTON package.
# Copyright (C) 2023-2024 The HORTON-PART Development Team
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
import argparse
import logging
import os
import subprocess
import sys

__all__ = ["main"]

logger = logging.getLogger(__name__)


def main(args=None) -> int:
    """
    Main program for the Horton-Part application.

    This program accepts a configuration file as input and executes
    the 'part-gen' and 'part-dens' commands with the given configuration.

    Parameters
    ----------
    args : list
        List of arguments passed from command line.

    Returns
    -------
    int :
        Exit status (0 for success, non-zero for errors).
    """

    # Program description
    description = "Horton-Part main program."

    # Argument parsing
    parser = argparse.ArgumentParser(prog="part", description=description)
    parser.add_argument("filename", type=str, help="User input cfg file.")
    parsed_args = parser.parse_args(args)

    # Ensuring the config file exists
    if not os.path.exists(parsed_args.filename):
        print(f"Error: The file '{parsed_args.filename}' does not exist.")
        return 1

    # Execute commands safely
    try:
        subprocess.run(["part-gen", parsed_args.filename], check=True)
        subprocess.run(["part-dens", parsed_args.filename], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing a command: {e}")
        return e.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
