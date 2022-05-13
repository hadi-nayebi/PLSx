from pathlib import Path


def get_root(file, retrace=0):
    """Get the root path."""
    return Path(file).resolve().parents[retrace]
