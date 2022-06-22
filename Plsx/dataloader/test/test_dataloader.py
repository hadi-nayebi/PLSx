#!/usr/bin/env python
# coding: utf-8

"""Unit test for DataLoader."""

from io import StringIO
from unittest import TestCase
from unittest import main as unittest_main
from unittest.mock import patch

from torch import Tensor, cuda, device

from Plsx.dataloader.dataloader import DataLoader


class TestDataLoader(TestCase):
    """Test items for DataLoader class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.dl = None
        cls.device = device("cuda:0" if cuda.is_available() else "cpu")
        cls.DATASET_CL = "cl_100"
        cls.DATASET_SS = "ss_100"
        cls.DATASET_CLSS = "clss_20"
        cls.LARGE_BATCH = 12
        cls.SMALL_BATCH = 4

    def test_load_data(self):
        """Test load_data."""
        self.dl = DataLoader()
        self.dl.load_data(self.DATASET_CLSS)
        self.assertEqual(len(self.dl.test_data), 20)

    def test_test_train_data(self):
        """Test transform_train_data."""
        self.dl = DataLoader()
        self.dl.test_data = self.DATASET_CL
        self.dl.train_data = self.DATASET_CL
        self.assertIsInstance(self.dl.test_data, dict)
        self.assertIsInstance(self.dl.train_data, dict)

    def test_transform_data(self):
        """Test transform_data."""
        self.dl = DataLoader()
        self.dl.test_data = self.DATASET_CL
        self.dl.train_data = self.DATASET_CL
        self.dl.transform_data(self.device)

    def test_get_by_key(self):
        """Test get_by_key."""
        self.dl = DataLoader()
        self.dl.test_data = self.DATASET_CL
        self.dl.train_data = self.DATASET_CL
        self.dl.transform_data(self.device)
        key = "CT694_01890"
        item = self.dl.get_by_key(key)
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 2)
        self.assertIsInstance(item[0], Tensor)
        self.assertIsInstance(item[1], dict)
        self.assertEqual(item[1]["name"], key)
        # train data
        key = "Bcen_0564"
        item = self.dl.get_by_key(key, dataset="train")
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 2)
        self.assertIsInstance(item[0], Tensor)
        self.assertIsInstance(item[1], dict)
        self.assertEqual(item[1]["name"], key)

    def test_get_all(self):
        """Test get_all."""
        self.dl = DataLoader()
        self.dl.test_data = self.DATASET_CL
        self.dl.train_data = self.DATASET_CL
        self.dl.transform_data(self.device)
        items = list(self.dl.get_all())
        self.assertEqual(len(items), len(self.dl.test_data) + len(self.dl.train_data))

    def test_get_train_batch(self):
        """Test get_train_batch."""
        self.dl = DataLoader()
        # yield None
        batches = [item for item, _ in zip(self.dl.get_train_batch(self.SMALL_BATCH), range(5))]
        self.assertIsNone(batches[-1])
        # yield batch
        self.dl.train_data = self.DATASET_CL
        self.dl.transform_data(self.device)
        batches = [item for item, _ in zip(self.dl.get_train_batch(self.SMALL_BATCH), range(5))]
        self.assertEqual(len(batches), 5)
        batches = [item for item, _ in zip(self.dl.get_train_batch(self.LARGE_BATCH), range(50))]
        self.assertEqual(len(batches), 50)

    def test_get_test_batch(self):
        """Test get_test_batch."""
        self.dl = DataLoader()
        self.dl.test_data = self.DATASET_CL
        self.dl.transform_data(self.device)
        items = list(self.dl.get_test_batch(test_items=["HQ_3387A", "GKZ92_08920"]))
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][1]["name"], "HQ_3387A")
        self.assertEqual(items[1][1]["name"], "GKZ92_08920")
        # prints Key does not exist in dataset
        with patch("sys.stdout", new=StringIO()) as fake_out:
            items = list(self.dl.get_test_batch(test_items=["INVALID_KEY"]))
            self.assertEqual(fake_out.getvalue(), "INVALID_KEY does not exist in dataset.\n")
        # num_test_items == -1
        items = list(self.dl.get_test_batch(num_test_items=-1))
        self.assertEqual(len(items), len(self.dl.test_data))


if __name__ == "__main__":
    unittest_main()
