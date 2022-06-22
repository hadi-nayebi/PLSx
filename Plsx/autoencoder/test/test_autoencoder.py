#!/usr/bin/env python
# coding: utf-8

"""Unit test for Autoencoder."""

from unittest import TestCase
from unittest import main as unittest_main

from torch import Tensor, all, cat, eq, rand, randint

from PLSx.autoencoder.architecture import Architecture
from PLSx.autoencoder.autoencoder import Autoencoder
from PLSx.utils.file_manager import get_root


class TestAutoencoder(TestCase):
    """Test items for Autoencoder class."""

    root = get_root(__file__, retrace=2)

    def test_init(self):
        """Test init."""
        architecture_json_file = self.root / "test_data" / "architecture_sample.json"
        architecture = Architecture()
        architecture.build(architecture_json_file)
        ae = Autoencoder(architecture=architecture)
        self.assertIsInstance(ae, Autoencoder)

    def test_transform_input(self):
        """Test transform_input."""
        architecture_json_file = self.root / "test_data" / "architecture_sample.json"
        architecture = Architecture()
        architecture.build(architecture_json_file)
        ae = Autoencoder(architecture=architecture)
        input_vals = cat(
            [randint(0, ae.d0, (200, 1)), randint(0, ae.ds, (200, 1)), rand((200, 1)),], dim=1,
        )
        # A--
        input_ndx, target_vals_ss, target_vals_cl, one_hot_input = ae.transform_input(
            input_vals, device=None, input_keys="A--"
        )
        self.assertIsInstance(input_ndx, Tensor)
        self.assertIsNone(target_vals_ss)
        self.assertIsNone(target_vals_cl)
        self.assertIsInstance(one_hot_input, Tensor)
        # AS-
        input_ndx, target_vals_ss, target_vals_cl, one_hot_input = ae.transform_input(
            input_vals, device=None, input_keys="AS-"
        )
        self.assertIsInstance(input_ndx, Tensor)
        self.assertIsInstance(target_vals_ss, Tensor)
        self.assertIsNone(target_vals_cl)
        self.assertIsInstance(one_hot_input, Tensor)
        # A-C
        input_ndx, target_vals_ss, target_vals_cl, one_hot_input = ae.transform_input(
            input_vals, device=None, input_keys="A-C"
        )
        self.assertIsInstance(input_ndx, Tensor)
        self.assertIsNone(target_vals_ss)
        self.assertIsInstance(target_vals_cl, Tensor)
        self.assertIsInstance(one_hot_input, Tensor)
        # ASC
        input_ndx, target_vals_ss, target_vals_cl, one_hot_input = ae.transform_input(
            input_vals, device=None, input_keys="ASC"
        )
        self.assertIsInstance(input_ndx, Tensor)
        self.assertIsInstance(target_vals_ss, Tensor)
        self.assertIsInstance(target_vals_cl, Tensor)
        self.assertIsInstance(one_hot_input, Tensor)

        self.assertEqual(input_ndx.shape, (181, ae.w))
        self.assertEqual(target_vals_ss.shape, (181, ae.w))
        self.assertEqual(target_vals_cl.shape, (181, 2))
        self.assertEqual(one_hot_input.shape, (181, ae.w, ae.d0))

        input_vals = cat(
            [randint(0, ae.d0, (200, 1)), rand((200, 1)), randint(0, ae.ds, (200, 1)),], dim=1,
        )
        # ACS
        input_ndx, target_vals_ss, target_vals_cl, one_hot_input = ae.transform_input(
            input_vals, device=None, input_keys="ACS"
        )
        self.assertIsInstance(input_ndx, Tensor)
        self.assertIsInstance(target_vals_ss, Tensor)
        self.assertIsInstance(target_vals_cl, Tensor)
        self.assertIsInstance(one_hot_input, Tensor)

        self.assertEqual(input_ndx.shape, (181, ae.w))
        self.assertEqual(target_vals_ss.shape, (181, ae.w))
        self.assertEqual(target_vals_cl.shape, (181, 2))
        self.assertEqual(one_hot_input.shape, (181, ae.w, ae.d0))

    def test_transform_noisy_input(self):
        """Test transform_input with noise."""
        architecture_json_file = self.root / "test_data" / "architecture_sample.json"
        architecture = Architecture()
        architecture.build(architecture_json_file)
        ae = Autoencoder(architecture=architecture)
        input_vals = cat(
            [randint(0, ae.d0, (200, 1)), randint(0, ae.ds, (200, 1)), rand((200, 1)),], dim=1,
        )
        # A--
        _, _, _, one_hot_input = ae.transform_input(input_vals, device=None, input_keys="A--")
        _, _, _, noisy_one_hot_input = ae.transform_input(
            input_vals, device=None, input_keys="A--", input_noise=0.02
        )
        self.assertEqual(one_hot_input.sum(), noisy_one_hot_input.sum())
        self.assertFalse(all(eq(one_hot_input, noisy_one_hot_input)))

    def test_forward(self):
        """Test autoencoder forward"""
        architecture_json_file = self.root / "test_data" / "architecture_sample.json"
        self.assertTrue(
            architecture_json_file.exists(), f"{architecture_json_file} does not exist."
        )
        architecture = Architecture()
        architecture.build(architecture_json_file)
        ae = Autoencoder(architecture=architecture)
        ae.architecture.get_model()
        input_vals = cat(
            [randint(0, ae.d0, (200, 1)), rand((200, 1)), randint(0, ae.ds, (200, 1)),], dim=1,
        )
        # ACS
        _, _, _, one_hot_input = ae.transform_input(input_vals, device=None, input_keys="ACS")
        output_vals = ae.forward(one_hot_input)
        self.assertEqual(output_vals["vectorizer"].shape, (181 * ae.w, ae.d1))
        self.assertEqual(output_vals["classifier"].shape, (181, 2))
        self.assertEqual(output_vals["encoder"].shape, (181, ae.dn))
        self.assertEqual(output_vals["decoder"].shape, (181 * ae.w, ae.d1))
        self.assertEqual(output_vals["ss_decoder"].shape, (181 * ae.w, ae.ds))
        self.assertEqual(output_vals["devectorizer"].shape, (181 * ae.w, ae.d0))


if __name__ == "__main__":
    unittest_main()
