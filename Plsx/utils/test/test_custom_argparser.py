#!/usr/bin/env python
# coding: utf-8

"""Unit test for ArgParser."""

import sys
from unittest import TestCase
from unittest import main as unittest_main

from PLSx.utils.custom_argparser import CustomArgParser, TrainSessionArgParser


class TestArgParser(TestCase):
    """Test items for ArgParser class."""

    def test_custom_arg_parser(self):
        """CustomArgParser object returns args as a dict."""
        parser = CustomArgParser()
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
        parser.add_argument("--name", "-n", type=str, help="Name", default="Dummy")
        args = parser.parse_args(["--verbose", "--name", "ISACC"])
        self.assertTrue(args.verbose)
        self.assertEqual(args.name, "ISACC")
        # dict
        sys.argv = [sys.argv[0], "--verbose", "--name=ISACC"]
        help_value_dict = parser.get_help_value_dict()
        self.assertIn("Verbose", help_value_dict)
        self.assertIn("Name", help_value_dict)
        self.assertEqual(help_value_dict["Name"], "ISACC")
        self.assertEqual(help_value_dict["Verbose"], True)

    def test_train_session_argparser(self):
        """TrainSessionArgParser object returns args as a dict."""
        parser = TrainSessionArgParser()
        sys.argv = [sys.argv[0], "-n=ISACC"]
        parsed_args = parser.parsed()
        self.assertIn("Model Name", parsed_args)
        # self.assertIn("dataset_ss", parsed_args)
        # self.assertIn("dataset_clss", parsed_args)
        # self.assertIn("arch", parsed_args)
        # self.assertIn("epochs", parsed_args)
        # self.assertIn("train_batch", parsed_args)
        # self.assertIn("test_batch", parsed_args)
        # self.assertIn("test_interval", parsed_args)


if __name__ == "__main__":
    unittest_main()
