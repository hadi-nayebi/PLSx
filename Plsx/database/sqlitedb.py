from os.path import dirname
from pathlib import Path

import sqlalchemy
from pandas import DataFrame

from PLSx.database.database import Database


class SqliteDB(Database):
    """SqliteDB subclass of Database abstract class

    Args:
        Database (ABC): abstract base class for database objects
    """

    root = Path(dirname(__file__)).parent.parent

    def __init__(self):
        super().__init__()
        self.engine = None

    def create(self, name: str) -> None:
        """Create a new database.

        Args:
            name (_type_): Name of the database

        Returns:
            None: returns None
        """
        data_dir = str(self.root / "data")
        self.table_name = name
        self.engine = sqlalchemy.create_engine(f"sqlite://{data_dir}/" + name + ".db")
