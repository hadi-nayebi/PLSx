#!/usr/bin/env python
# coding: utf-8

"""Unit test for Architecture."""
from os.path import dirname
from pathlib import Path
from unittest import TestCase
from unittest import main as unittest_main

from PLSx.autoencoder.architecture import Architecture


class TestArchitecture(TestCase):
    """Test items for Architecture class."""

    root = Path(dirname(__file__)).parent.parent.parent

    def test_architecture(self):
        """Test architecture."""
        architecture_json_file = (
            self.root / "config" / "architecture_sample.json"
        )  # TODO: include all types of layers in the file
        assert architecture_json_file.exists(), f"{architecture_json_file} does not exist."
        architecture = Architecture()
        architecture.build(architecture_json_file)
        model = architecture.get_model()


if __name__ == "__main__":
    unittest_main()
