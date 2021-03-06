import sqlite3
from os.path import dirname
from pathlib import Path

from pandas import DataFrame

from Plsx.database.database import Database


class SqliteDB(Database):
    """SqliteDB subclass of Database abstract class

    Args:
        Database (ABC): abstract base class for database objects
    """

    root = Path(dirname(__file__)).parent.parent

    def __init__(self):
        super().__init__()
        self.engine = None
        self.table_name = None
        self.db_dir = None

    def create(self, name: str) -> None:
        """Create a new database.

        Args:
            name (_type_): Name of the database

        Returns:
            None: returns None
        """
        self.db_dir = str(self.root / "data" / "db")
        self.table_name = name
        self.engine = sqlite3.connect(self.db_dir / f"{name}.db")
