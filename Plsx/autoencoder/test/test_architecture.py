#!/usr/bin/env python
# coding: utf-8

"""Unit test for Architecture."""
from os.path import dirname
from pathlib import Path
from unittest import TestCase
from unittest import main as unittest_main

from torch.nn import Sequential

from PLSx.autoencoder.architecture import Architecture, Layer, Unit
from PLSx.dataloader.utils import read_json


class TestArchitecture(TestCase):
    """Test items for Architecture class."""

    root = Path(dirname(__file__)).parent.parent

    def test_architecture(self):
        """Test architecture."""
        architecture_json_file = (
            self.root / "test_data" / "architecture_sample.json"
        )  # TODO: include all types of layers in the file
        self.assertTrue(
            architecture_json_file.exists(), f"{architecture_json_file} does not exist."
        )
        architecture = Architecture()
        architecture.build(architecture_json_file)
        # raise warning if units are built already
        keys = list(architecture.components.keys())
        # config = read_json(architecture_json_file)["components"][keys[0]]
        # with self.assertWarns(Warning):
        #     architecture.components[keys[0]].build(config)
        model = architecture.get_model()
        self.assertIsInstance(model, dict)
        self.assertEqual(len(model), len(architecture.components))
        self.assertIsInstance(model[keys[0]], Sequential)
        # test with str
        architecture = Architecture()
        architecture.build(str(architecture_json_file))

    def test_architecture_errors(self):
        """Test architecture errors."""
        architecture_json_file = (
            self.root / "test_data" / "architecture_sample_invalide_activation.json"
        )
        self.assertTrue(
            architecture_json_file.exists(), f"{architecture_json_file} does not exist."
        )
        architecture = Architecture()
        architecture.build(architecture_json_file)
        self.assertRaises(ValueError, architecture.get_model)
        self.assertRaises(ValueError, architecture.build_units, "invalid", [{"dummy": "dummy"}])

    def test_unit_errors(self):
        """Test Unit."""
        unit = Unit()
        self.assertRaises(ValueError, unit.build, [])
        self.assertRaises(ValueError, unit.layer_maker, {"type": "invalid"})

    def test_notImplemented(self):
        """Test not Implemented errors."""
        layer = Layer(type="dummy")
        self.assertRaises(NotImplementedError, layer.make)

        unit = Unit()
        self.assertRaises(NotImplementedError, unit.add_layer, layer)
        self.assertRaises(NotImplementedError, unit.insert_layer, layer, 1)


if __name__ == "__main__":
    unittest_main()
