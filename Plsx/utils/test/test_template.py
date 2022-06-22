#!/usr/bin/env python
# coding: utf-8

"""Unit test for XXX."""

from unittest import TestCase
from unittest import main as unittest_main

from PLSx.utils.file_manager import get_root


class TestXXX(TestCase):
    """Test items for XXX class."""

    root = get_root(__file__, retrace=3)


if __name__ == "__main__":
    unittest_main()
