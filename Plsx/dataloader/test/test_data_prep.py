#!/usr/bin/env python
# coding: utf-8

"""Unit test for DataPrep."""

import sys

sys.path.insert(1, get_root(__file__, retrace=3))
from os import system
from os.path import dirname
from pathlib import Path
from unittest import TestCase
from unittest import main as unittest_main

from PLSx.dataloader.data_prep import DataPrep
from PLSx.utils.file_manager import get_root


class TestDataPrep(TestCase):
    """Test items for DataPrep."""

    root = get_root(__file__, retrace=3)

    def test_load_source(self):
        """Test load_source."""
        data_prep = DataPrep()
        source = self.root / "data" / "uniprot" / "UP000000212" / "UP000000212_1234679.xml"
        source = self.root / "data" / "uniprot" / "UP000000212" / "UP000000212_1234679.xml"
        data_prep.load_source(source)


if __name__ == "__main__":
    unittest_main()
