"""Defines classes for custom argparser objects for differennt interfaces."""

from argparse import ArgumentParser
from typing import Any, Dict

HelpValuePair = Dict[str, Any]


class CustomArgParser(ArgumentParser):
    """Add help_value_pairs method to ArgumentParser object."""

    def help_value_pairs(self) -> HelpValuePair:
        """Create and return dict of {option_help_message: option_value}."""
        parsed_args = self.parse_args()
        help_value_pair_dict = {}
        for value in self.__dict__["_option_string_actions"].values():
            if value.dest in parsed_args.__dict__.keys():
                help_value_pair_dict[value.help] = parsed_args.__dict__[value.dest]
        return help_value_pair_dict


class DefaultParser(object):
    """DefaultParser is the basic parser class. More specialized arg parsers will inherit from this class."""

    def __init__(self, desc: str) -> None:
        """Define instance variables, collect arguments."""
        self.parser = CustomArgParser(description=desc)
        self._initialize()

    def _initialize(self) -> None:
        """A virtual method."""

    def parsed(self) -> HelpValuePair:
        """Return description-value pair dictionary."""
        return self.parser.help_value_pairs()


class TrainSessionArgParser(DefaultParser):
    """Set description, options, and flags for TrainSession."""

    def __init__(self) -> None:
        """Define instance variables, collect arguments."""
        super().__init__("Train a protein sequence autoencoder")

    def _initialize(self) -> None:
        self.parser.add_argument("-n", "--model", type=str, help="Model Name", required=True)
        self.parser.add_argument("-dss", "--dataset_ss", type=str, help="Dataset_ss", default="")
        self.parser.add_argument(
            "-dclss", "--dataset_clss", type=str, help="Dataset_clss", default=""
        )
        # architecture blueprint
        self.parser.add_argument("-a", "--arch", type=str, help="Arch", required=True)
        self.parser.add_argument("-e", "--epochs", type=int, help="Epochs", default=25)
        self.parser.add_argument("-trb", "--train_batch", type=int, help="Train Batch", default=128)
        self.parser.add_argument("-teb", "--test_batch", type=int, help="Test Batch", default=1)
        self.parser.add_argument(
            "-ti", "--test_interval", type=int, help="Test Interval", default=100
        )
        self.parser.add_argument("-no", "--noise", type=float, help="Input Noise", default=0.0)
        self.parser.add_argument(
            "-mvid", "--model_version_id", type=str, help="Model Version ID", default=""
        )
        self.parser.add_argument(
            "-ts", "--training_settings", type=str, help="Training Settings", default=None,
        )
        self.parser.add_argument(
            "-mts",
            "--modular_training_settings",
            type=str,
            help="Modular Training Settings",
            default=None,
        )
        self.parser.add_argument(
            "-nt", "--no_train", help="No Train", action="store_true", default=False
        )
        self.parser.add_argument(
            "-it", "--is_testing", help="Is Testing", action="store_true", default=False
        )
        self.parser.add_argument(
            "-of", "--overfitting", help="Overfitting", action="store_true", default=False,
        )
        self.parser.add_argument(
            "-v", "--verbose", help="Verbose", action="store_true", default=False
        )
        self.parser.add_argument("-le", "--log_every", type=int, help="Log every", default=100)
        self.parser.add_argument(
            "-nc", "--no_continuity", help="No Continuity", action="store_true", default=False,
        )
        self.parser.add_argument("-s", "--seed", type=int, help="Random Seed", default=0)
        self.parser.add_argument(
            "-smi", "--save_model_interval", type=int, help="Save Model Interval", default=1,
        )
        self.parser.add_argument("-b", "--branch", type=str, help="Branch", default="")
        self.parser.add_argument("-f", "--focus", type=str, help="Focus", default=None)
