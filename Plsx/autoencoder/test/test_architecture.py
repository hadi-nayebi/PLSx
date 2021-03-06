#!/usr/bin/env python
# coding: utf-8

"""Unit test for Architecture."""
from unittest import TestCase
from unittest import main as unittest_main

from torch.nn import Module, Sequential

from PLSx.autoencoder.architecture import Architecture, Layer, Unit
from PLSx.dataloader.utils import read_json
from PLSx.utils.file_manager import get_root


class TestArchitecture(TestCase):
    """Test items for Architecture class."""

    root = get_root(__file__, retrace=2)

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
        architecture.get_model()
        self.assertTrue(architecture.is_built)
        self.assertTrue(
            all([isinstance(architecture.components[key].unit, Sequential) for key in keys])
        )
        # test model reference
        self.assertEqual(
            list(architecture.components[keys[0]].unit.children())[0],
            architecture.components[keys[0]].layers[0].layer,
        )
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

    def test_params(self):
        """Test architecture errors."""
        architecture_json_file = self.root / "test_data" / "architecture_sample.json"
        self.assertTrue(
            architecture_json_file.exists(), f"{architecture_json_file} does not exist."
        )
        architecture = Architecture()
        architecture.build(architecture_json_file)
        self.assertTrue(architecture.d0, 21)
        self.assertTrue(architecture.d1, 8)
        self.assertTrue(architecture.dn, 10)
        self.assertTrue(architecture.w, 20)
        self.assertTrue(architecture.ds, 9)


if __name__ == "__main__":
    unittest_main()
