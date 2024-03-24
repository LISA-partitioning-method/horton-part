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

import yaml

from horton_part import PERIODIC_TABLE, __version__, setup_logger

__all__ = ["PartProg", "load_settings_from_yaml_file"]


def load_settings_from_yaml_file(args, sub_cmd="part-gen", fn_key="config_file"):
    """
    Load settings from a YAML configuration file and update the 'args' object.

    This function reads a YAML file specified by the 'fn_key' attribute of the 'args' object.
    It then updates 'args' with the settings found under the specified 'cmd' section of the YAML file.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments object, typically obtained from argparse. This object is updated
        with the settings from the YAML file.
    sub_cmd : str, optional
        The key within the YAML file that contains the settings to be loaded. Only settings
        under this key are used to update 'args'. Default is 'part-gen'.
    fn_key : str, optional
        The attribute name in 'args' that holds the path to the YAML configuration file.
        Default is 'config_file'.

    Returns
    -------
    argparse.Namespace
        The updated arguments object with settings loaded from the YAML file.

    Raises
    ------
    AssertionError
        If 'args' does not have an attribute named 'fn_key', or if the 'cmd' key is not
        found in the loaded YAML settings.

    Notes
    -----
    - The function asserts that the 'args' object has an attribute named as per 'fn_key'.
    - It also checks if the specified YAML file exists before attempting to open it.
    - Settings are loaded only if the 'cmd' key is present in the YAML file.
    - Each setting under the 'cmd' section in the YAML file updates the corresponding attribute
      in the 'args' object.
    """
    assert hasattr(args, fn_key)
    yaml_fn = getattr(args, fn_key)
    if yaml_fn and os.path.exists(yaml_fn):
        with open(yaml_fn) as f:
            settings = yaml.safe_load(f)
        if sub_cmd:
            assert sub_cmd in settings
            for k, v in settings[sub_cmd].items():
                setattr(args, k, v)
        else:
            for k, v in settings.items():
                setattr(args, k, v)
    return args


class PartProg:
    def __init__(self, program_name, width):
        self.width = width
        self.program_name = program_name
        self.logger = logging.getLogger(program_name)

    def run(self, args=None) -> int:
        """Main entry."""
        parser = self.build_parser()
        args = parser.parse_args(args)
        args = load_settings_from_yaml_file(args, self.program_name)
        if args.inputs is None:
            parser.print_help()
            return 0

        log_files = getattr(args, "log_files", None)
        inputs = args.inputs
        outputs = args.outputs
        if not isinstance(inputs, list):
            inputs = [inputs]
            outputs = [outputs]
        if log_files is None:
            log_files = [None] * len(inputs)
        assert len(inputs) == len(outputs) == len(log_files)
        for fn_in, fn_out, fn_log in zip(inputs, outputs, log_files):
            self.single_launch(args, fn_in, fn_out, fn_log)
        return 0

    def setup_logger(self, args: argparse.Namespace, fn_log, **kwargs):
        # Convert the log level string to a logging level
        if not hasattr(args, "log_level"):
            log_level = logging.INFO
        else:
            log_level = getattr(logging, args.log_level, logging.INFO)
        setup_logger(self.logger, log_level, fn_log, **kwargs)

    def single_launch(self, *args, **kwargs):
        """Man entry for a single job."""
        raise NotImplementedError

    def build_parser(self, *args, **kwargs):
        """Parse command-line arguments."""
        raise NotImplementedError

    def print_settings(self, args, fn_in, fn_out, fn_log, exclude_keys=None):
        """Print setting for this program."""
        self.print_header(
            f"Settings for {self.program_name} program with Horton-Part {__version__}"
        )
        for k, v in vars(args).items():
            if exclude_keys and k in exclude_keys:
                continue
            if k in ["inputs", "outputs", "log_files"]:
                if k == "inputs":
                    self.logger.info(f"{'input':>40} : {str(fn_in):<40}".center(self.width, " "))
                elif k == "outputs":
                    self.logger.info(f"{'output':>40} : {str(fn_out):<40}".center(self.width, " "))
                else:
                    self.logger.info(
                        f"{'log_file':>40} : {str(fn_log):<40}".center(self.width, " ")
                    )

            else:
                self.logger.info(f"{k:>40} : {str(v):<40}".center(self.width, " "))
        self.print_line()

    def print_header(self, header):
        self.logger.info("*" * self.width)
        self.logger.info(f" {header} ".center(self.width, " "))
        self.logger.info("*" * self.width)

    def print_line(self):
        self.logger.info("-" * self.width)
        self.logger.info(" ")

    def print_coordinates(self, numbers, coordinates):
        self.logger.info("Coordinates [a.u.] : ")
        self.logger.info(f"{' ':>10} {'X':>15} {'Y':>15} {'Z':>15}".center(self.width, " "))
        for number, xyz in zip(numbers, coordinates):
            self.logger.info(
                f"{PERIODIC_TABLE[number]:>10} {xyz[0]:>15.5f} {xyz[1]:>15.5f} {xyz[2]:>15.5f}".center(
                    self.width, " "
                )
            )
        self.print_line()

    def print_charges(self, numbers, charges):
        self.logger.info("  Atomic charges [a.u.]:")
        # for atnum, chg in zip(data["atnums"], part.cache["charges"]):
        for atnum, chg in zip(numbers, charges):
            self.logger.info(f"{PERIODIC_TABLE[atnum]:>4} : {chg:>15.8f}".center(self.width, " "))
        self.logger.info("-" * self.width)
        self.logger.info(" " * self.width)
