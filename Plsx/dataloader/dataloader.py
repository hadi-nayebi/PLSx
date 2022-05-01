"""Defines the DataLoader class and related I/O methods."""

from os.path import dirname
from pathlib import Path


class DataLoader(object):
    """Dataloader class maintains the training and testing data for the model."""

    root = Path(dirname(__file__)).parent.parent
