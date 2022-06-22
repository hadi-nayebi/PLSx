#!/usr/bin/env python
# coding: utf-8

"""Unit test for DataPrep."""

import sys
from pathlib import Path

# sys.path.insert(1, Path(__file__).resolve().parents[3])
# print(Path(__file__).resolve().parents[3])
# print(sys.path)
from unittest import TestCase
from unittest import main as unittest_main

from Plsx.dataloader.data_prep import DataPrep
from Plsx.utils.file_manager import get_root


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
