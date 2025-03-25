""" Multirun orchestration utility classes. """
import importlib
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from sapicore.pipeline import Pipeline
from sapicore.utils.io import ensure_dir

__all__ = ("ConfigEditor", "MultiRun", "parse_arguments")


class ConfigEditor:
    """Load or edit a specific nested value in a YAML file or a dictionary."""

    def __init__(self, path: str = None, content: dict = None):
        self.path = path
        self.content = self.load() if (path and content is None) else content

    def load(self):
        """Returns content of YAML at `path`."""
        return YAML().load(Path(self.path))

    def edit(self, destination: list, entry: Any):
        """Edits YAML in `path`, replacing the nested value in `destination` with `entry`."""
        self._touch(self.content, destination, entry)
        if self.path is not None:
            YAML().dump(self.content, Path(self.path))

    @staticmethod
    def _touch(d: dict, lst: list, entry: Any = None) -> dict:
        """Returns the dictionary `d` after replacing the value at `lst` with `entry`."""
        if len(lst) == 1:
            if entry is not None:
                d[lst[0]] = entry
            return d
        return ConfigEditor._touch(d[lst[0]], lst[1:], entry)


class MultiRun(Pipeline):
    """Chains parameterized runs of the same simulation pipeline.

    The configuration dictionary specifies which YAMLs to touch, what parameters to change, and in what order.

    When a key is identified as a path to a YAML file, the value is treated as a set of editing instructions;
    otherwise, it is treated as the name of an organization level containing such instructions.

    Thus, the format supports nested experimental designs, where additional levels of nesting can be
    encountered after a YAML has been touched.

    Warning
    -------
    This version modifies YAML files on disk as necessary. For replication purposes, all configuration YAMLs
    in the project directory tree are zipped and saved alongside the results of each branch.

    """

    def __init__(self, config: dict | str = None, out_dir: str = None, child_kwargs: dict = None):
        # base pipeline constructor.
        super().__init__(config_or_path=config)

        # pipeline class to chain multiple instances of.
        self.pipeline = self._pipeline_import()
        self.arguments = child_kwargs if child_kwargs is not None else {}

        self.run_dir = out_dir

        # where we currently are in the traversal.
        self.pointer = self.configuration.get("grid")

        # for tracking concurrent processes.
        self.futures = []

    def _pipeline_import(self) -> [Pipeline]:
        """Imports the pipeline object to be run with each parameter configuration."""
        module = importlib.import_module(self.configuration.get("package"))
        return getattr(module, self.configuration.get("pipeline"))

    @staticmethod
    def _scan_levels(instructions: dict) -> int:
        """Scans a dictionary for its number of levels, validating its format along the way.

        Note
        ----
        List entries must have the same number of values ("zipped" traversal).

        """
        levels = 1
        for yaml_file, order in instructions.items():
            for destination, value in order.items():
                if isinstance(value, list):
                    if len(value) != levels and levels > 1:
                        raise ValueError(f"Misconfigured experimental design at {destination}: {value} != {levels}")
                    levels = len(value)
        return levels

    @staticmethod
    def _parse_level(level: dict) -> (dict, dict, list[str]):
        """Splits a given dictionary into its instructions component and inner conditions component."""
        instructions = {k: v for k, v in level.items() if os.path.exists(k)}
        inner_conditions = {k: v for k, v in level.items() if k != "names" and not os.path.exists(k)}
        names = level.get("names", "")

        return instructions, inner_conditions, names

    def _process_entries(self, nested_dict: dict, path: str):
        """Recursively processes nested dictionary entries describing tasks contingent on configuration editing jobs.

        Each level of the dictionary contains (1) path keys, (2) condition keys. Path keys specify YAML files,
        and their values are a set of editing instructions. Editing instructions may map multiple values to the
        same YAML field, meaning they should be looped over (the number of values should match across entries,
        as in "zipped" traversal).

        If the dictionary contains condition keys, a recursive call to this method is made with a
        sub-dictionary containing them and an updated output directory. If it only contains path keys,
        the child pipeline's run() method is invoked.

        Thus, the leaves of this nested structure are executed with every requested parameter combination above it,
        reflecting the nested experimental design specified by the multirun YAML.

        """
        instructions, inner_dict, level_names = self._parse_level(nested_dict)

        for k in range(self._scan_levels(instructions)):
            if not instructions:
                # this is a container condition, with no jobs to perform.
                if inner_dict:
                    for key in inner_dict.keys():
                        # in this case, subdirectory names are the sub-condition (key) names.
                        self._process_entries(inner_dict[key], ensure_dir(os.path.join(path, key)))
            else:
                for yaml_path, order in instructions.items():
                    if os.path.exists(yaml_path):
                        editor = ConfigEditor(path=yaml_path)

                        # edit YAML values as necessary.
                        for destination, value in order.items():
                            if not isinstance(value, list):
                                editor.edit(destination, value)
                            else:
                                # if multiple entries are list-formatted, parameters are zipped.
                                editor.edit(destination, value[k])

                # having finished editing the parent branch's parameters, call on child branches.
                if inner_dict:
                    for key in inner_dict.keys():
                        self._process_entries(inner_dict[key], ensure_dir(os.path.join(path, key)))
                else:
                    # execute the pipeline with the current parameterization.
                    task = self.pipeline(**self.arguments)
                    task.out_dir = ensure_dir(os.path.join(path, level_names[k]))

                    task.run()

    def run(self):
        self._process_entries(self.configuration.get("grid"), self.run_dir)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "-experiment",
        action="store",
        dest="experiment",
        required=False,
        default="yaml/synthetic.yaml",
        help="Path to an experiment configuration YAML.",
    )
    parser.add_argument(
        "-multirun",
        action="store",
        dest="multirun",
        required=False,
        default=False,
        help="Path to a multirun configuration YAML.",
    )
    parser.add_argument(
        "-data",
        action="store",
        dest="data",
        required=False,
        default="../../datasets",
        help="Destination path for fetched or generated datasets.",
    )
    parser.add_argument(
        "-output",
        action="store",
        dest="output",
        required=False,
        default="../../results",
        help="Destination path for trained model files (PT) and intermediate simulation output.",
    )
    return parser.parse_args()
