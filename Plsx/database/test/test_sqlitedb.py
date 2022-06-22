#!/usr/bin/env python
# coding: utf-8

"""Unit test for SqliteDB."""
from unittest import TestCase
from unittest import main as unittest_main

from sqlalchemy.engine.base import Engine

from Plsx.database.sqlitedb import SqliteDB


class TestSqliteDB(TestCase):
    """Test items for SqliteDB class."""

    def test_create(self):
        """Test create."""
        db = SqliteDB()
        db.create("test")
        self.assertIsInstance(db.engine, Engine)
        self.assertEqual(db.table_name, "test")
        db.engine.dispose()


if __name__ == "__main__":
    unittest_main()
