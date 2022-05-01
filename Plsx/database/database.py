from abc import ABC, abstractmethod


class Database(ABC):
    """Abstract base class for a database object that will store the data streamed from Pipeline objects

    Args:
        ABC (_type_): ABC base class
    """

    def create(self, name: str) -> None:
        """Create a new database.

        Args:
            name (_type_): Name of the database

        Returns:
            None: returns None
        """
