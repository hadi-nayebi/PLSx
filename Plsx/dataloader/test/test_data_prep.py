#!/usr/bin/env python
# coding: utf-8

"""Unit test for DataPrep."""
from pathlib import Path
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
        # source = (
        #     self.root / "data" / "uniprot" / "UP000000212" / "UP000000212_1234679.xml"
        # )
        source = self.root / "data" / "uniprot" / "uniprot_sprot.xml"
        if not source.exists():
            self.skipTest("Source file not found.")
        data_prep.load_source(source)
        self.assertTrue(data_prep.source)


if __name__ == "__main__":
    unittest_main()
