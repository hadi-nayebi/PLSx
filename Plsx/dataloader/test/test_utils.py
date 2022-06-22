#!/usr/bin/env python
# coding: utf-8

"""Unit test for DataLoader Utils."""

from os import system
from os.path import dirname
from pathlib import Path
from unittest import TestCase
from unittest import main as unittest_main

from numpy import array

from PLSx.dataloader.utils import (
    get_all_pfam_ids,
    read_fasta,
    read_json,
    write_fasta,
    write_json,
)


class TestUtils(TestCase):
    """Test items for DataLoader utils."""

    root = Path(dirname(__file__)).parent.parent

    def test_read_fasta(self):
        """Test read_fasta."""
        filename = self.root / "test_data" / "test_read_fasta.fasta"
        data = read_fasta(filename)
        self.assertEqual(len(data), 5)
        self.assertIsInstance(data, dict)

    def test_write_fasta(self):
        """Test write_fasta."""
        filename = self.root / "test_data" / "test_write_fasta.fasta"
        data = {
            "UPI00005627DD_0": "QSMSPELMAGDYVFCTVNGALSDYLSLEPIATFREPEGLTLVLEAEKAQ",
            "UPI00005627DD_1": "ESSALFSLITLTVHSSLEAVGLTAAFATKLAEHGISANVIAGYYHDHIFVQKEKAQQALQALG",
        }
        write_fasta(data, filename)
        self.assertTrue(filename.exists())
        system(f"rm {str(filename)}")

    def test_write_json(self):
        """Test write_json."""
        filename = self.root / "test_data" / "test_write_json.json"
        data = {
            "UPI00005627DD_0": "QSMSPELMAGDYVFCTVNGALSDYLSLEPIATFREPEGLTLVLEAEKAQ",
            "UPI00005627DD_1": "ESSALFSLITLTVHSSLEAVGLTAAFATKLAEHGISANVIAGYYHDHIFVQKEKAQQALQALG",
        }
        write_json(data, filename)
        self.assertTrue(filename.exists())
        system(f"rm {str(filename)}")
        # write .json.gz
        filename = self.root / "test_data" / "test_write_json.json.gz"
        write_json(data, filename)
        self.assertTrue(filename.exists())
        system(f"rm {str(filename)}")
        # invalid file format
        filename = self.root / "test_data" / "test_write_json.fasta"
        with self.assertRaises(IOError):
            write_json(data, filename)
        # using numpy.array, no encoding, .json
        filename = self.root / "test_data" / "test_write_json.json"
        data = {"item1": array([1, 2, 3]), "item2": array([4, 5, 6])}
        with self.assertRaises(TypeError):
            write_json(data, filename)
        # using numpy.array, .json.gz
        filename = self.root / "test_data" / "test_write_json.json.gz"
        write_json(data, filename, encoding="numpy")
        self.assertTrue(filename.exists())
        system(f"rm {str(filename)}")
        # using numpy.array, .json.gz no encoding
        filename = self.root / "test_data" / "test_write_json.json.gz"
        with self.assertRaises(TypeError):
            write_json(data, filename)
        # invalid file format
        filename = self.root / "test_data" / "test_write_json.fasta"
        with self.assertRaises(IOError):
            write_json(data, filename)
        # using str for data
        filename = self.root / "test_data" / "test_write_json.json"
        data = '{"item1": [1, 2, 3], "item2": [4, 5, 6]}'
        write_json(data, filename)
        self.assertTrue(filename.exists())
        system(f"rm {str(filename)}")
        # using not valid encoding
        filename = self.root / "test_data" / "test_write_json.json"
        with self.assertRaises(NotImplementedError):
            write_json(data, filename, encoding="invalid")
        # default encoding for non numpy array
        filename = self.root / "test_data" / "test_write_json.json"
        write_json(data, filename, encoding="numpy")
        self.assertTrue(filename.exists())
        system(f"rm {str(filename)}")

    def test_read_json(self):
        """Test read_json."""
        filename = self.root / "test_data" / "test_read_json.json"
        data = read_json(filename)
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data, dict)
        # read .json.gz
        filename = self.root / "test_data" / "test_read_json.json.gz"
        data = read_json(filename)
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data, dict)
        # invalid file format
        filename = self.root / "test_data" / "test_read_json.fasta"
        data = {
            "UPI00005627DD_0": "QSMSPELMAGDYVFCTVNGALSDYLSLEPIATFREPEGLTLVLEAEKAQ",
            "UPI00005627DD_1": "ESSALFSLITLTVHSSLEAVGLTAAFATKLAEHGISANVIAGYYHDHIFVQKEKAQQALQALG",
        }
        with self.assertRaises(IOError):
            read_json(filename)

    def test_get_all_pfam_ids(self):
        data = get_all_pfam_ids()


if __name__ == "__main__":
    unittest_main()
