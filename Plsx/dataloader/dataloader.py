"""Defines the DataLoader class and related I/O methods."""

from itertools import cycle
from os.path import dirname
from pathlib import Path
from typing import Tuple, Union

from numpy import arange, array
from numpy.random import choice, permutation
from torch import cat, device, tensor

from PLSx.dataloader.utils import read_json


class DataLoader(object):
    """Dataloader class maintains the training and testing data for the model."""

    root = Path(dirname(__file__)).parent.parent

    def __init__(self) -> None:
        """Initialize the DataLoader."""
        self._train_data = None
        self._test_data = None
        self.test_data_keys = None
        self.test_data_transform = False
        self.train_data_transform = False

    @property
    def test_data(self) -> dict:
        """Return the test data."""
        return self._test_data

    @test_data.setter
    def test_data(self, test_data: str) -> None:
        """Set the test data."""
        assert isinstance(test_data, str)
        test_data = self.root / "data" / "plsx" / f"{test_data}_test.json.gz"
        assert test_data.exists(), f"{test_data} does not exist."
        self._test_data = read_json(test_data)

    @property
    def train_data(self) -> dict:
        """Return the train data."""
        return self._train_data

    @train_data.setter
    def train_data(self, train_data: str) -> None:
        """Set the train data."""
        assert isinstance(train_data, str)
        train_data = self.root / "data" / "plsx" / f"{train_data}_train.json.gz"
        assert train_data.exists(), f"{train_data} does not exist."
        self._train_data = read_json(train_data)

    def transform_test_data(self, device: device) -> None:
        """Transform the test data into torch tensor."""
        if not self.test_data_transform:
            for key in self._test_data.keys():
                self._test_data[key] = self.to_tensor(self._test_data[key], key, device)
            self.test_data_keys = permutation(list(self._test_data.keys()))
            self.test_data_transform = True

    def transform_train_data(self, device: device) -> None:
        """Transform the train data into torch tensor."""
        if not self.train_data_transform:
            for key in self._train_data.keys():
                self._train_data[key] = self.to_tensor(self._train_data[key], key, device)
            self.train_data_transform = True

    def load_data(self, data: str) -> None:
        """Load the data."""
        assert isinstance(data, str)
        data = self.root / "data" / "plsx" / f"{data}.json.gz"
        assert data.exists(), f"{data} does not exist."
        self._test_data = read_json(data)

    @staticmethod
    def to_tensor(data: dict[str, array], key: str, device: device) -> tuple[tensor, dict]:
        """Transform the data into torch tensor."""
        output = None
        metadata = {"name": key}
        for i, (k, v) in enumerate(data.items()):
            if output is None:
                output = tensor(v, device=device).reshape((-1, 1))
                metadata[f"{i}"] = {"name": k, "shape": len(v)}
            else:
                output = cat((output, tensor(v, device=device).reshape((-1, 1))), dim=1)
                metadata[f"{i}"] = {"name": k, "shape": len(v)}
        assert output is not None, "Output is None."
        return output, metadata

    @staticmethod
    def join(items: tensor) -> tensor:
        return cat(items, dim=0)

    def transform_data(self, device: device) -> None:
        """Transform the data into torch tensor."""
        if self._test_data is not None:
            self.transform_test_data(device)
        if self._train_data is not None:
            self.transform_train_data(device)

    def get_by_key(self, key: str, dataset: str = "test") -> tuple[tensor, dict]:
        if dataset == "test":
            assert self._test_data is not None, "Test data is None."
            assert key in self._test_data.keys(), f"{key} does not exist."
            return self._test_data[key]
        elif dataset == "train":
            assert self._train_data is not None, "Train data is None."
            assert key in self._train_data.keys(), f"{key} does not exist."
            return self._train_data[key]

    def get_all(self) -> tuple[tensor, dict]:
        """Return all the data."""
        assert self._test_data is not None, "Test data is None."
        assert self._train_data is not None, "Train data is None."
        all_data = self._test_data.copy()
        all_data.update(self._train_data)
        for key in all_data.keys():
            yield all_data[key]

    def get_train_batch(self, batch_size: int = 128) -> tensor:
        """Return a batch of train data. For zipped datasets, if this dataset is not loaded, yield None.
        When zipped with larger datasets, this dataset cycles to match larger dataset."""
        num_batches = 10
        if self._train_data is not None:
            keys = permutation(list(self._train_data.keys()))
            num_batches = len(keys) // batch_size

        for i in cycle(range(num_batches)):
            if self._train_data is not None:
                yield self.join(
                    [
                        self._train_data[key][0]
                        for key in keys[i * batch_size : (i + 1) * batch_size]
                    ]
                )
            else:
                yield None

    def get_test_batch(
        self, num_test_items: int = 1, test_items: list[str] = None
    ) -> tuple[tensor, dict]:
        """Return a batch of test data."""
        if test_items is not None:
            for key in test_items:
                try:
                    assert key in self._test_data.keys()
                    yield self._test_data[key]
                except AssertionError:
                    print(f"{key} does not exist in dataset.")
        else:
            if num_test_items == -1:
                num_test_items = len(self._test_data.keys())
            num_test_items = min(num_test_items, len(self._test_data.keys()))
            ndx = choice(arange(len(self._test_data.keys())), size=num_test_items, replace=False)
            for i in ndx:
                yield self._test_data[self.test_data_keys[i]]
