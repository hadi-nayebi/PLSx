"""Helper functions for data loading."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from numpy import ndarray
from sqlalchemy import union

from Plsx.utils.file_manager import get_root


class NumpyEncoder(json.JSONEncoder):
    """Encoder for numpy arrays."""

    def default(self, obj) -> Any:
        if isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)  # no test coverage for this function


def read_fasta(filename: Union[Path, str]) -> Dict[str, str]:
    """Read fasta files and return a dict. (.fasta)"""
    data_dict = {}
    assert str(filename).endswith(".fasta"), "File format must be .fasta"
    with open(filename, "r") as file:
        for line in file.readlines():
            if line.startswith(">"):
                key = line.strip()[1:]
                data_dict[key] = ""
            else:
                data_dict[key] += line.strip()
    return data_dict


def write_fasta(data: dict, filename: Union[Path, str], line_size: int = 60) -> None:
    """Write fasta files from a dict. (.fasta)"""
    assert str(filename).endswith(".fasta"), "File format must be .fasta"
    with open(filename, "w") as file:
        for key, item in data.items():
            file.write(f">{key}\n")
            seq = item[:]
            while len(seq) > line_size:
                file.write(f"{seq[:line_size]}\n")
                seq = seq[line_size:]
            file.write(f"{seq}\n")


def read_json(filename: Union[Path, str]) -> Dict[str, Any]:
    """Read json files and return a dict. (.json, .json.gz)"""
    if isinstance(filename, Path):
        filename = str(filename)
    if filename.endswith(".json.gz"):
        with gzip.open(filename, "r") as file:
            json_bytes = file.read()
            json_str = json_bytes.decode("utf-8")
            return json.loads(json_str)
    elif filename.endswith(".json"):
        with open(filename, "r") as file:
            return json.load(file)
    else:
        raise IOError("File format must be .gz or .json.gz")


def write_json(
    data: Union[dict, str], filename: Union[Path, str], encoding=None, pretty=False
) -> None:
    """Write json file from a dict, encoding numpy arrays. (.json, .json.gz)"""
    if isinstance(filename, Path):
        filename = str(filename)
    # stting up the encoding
    if encoding is not None:
        if encoding == "numpy":
            encoding = NumpyEncoder
        else:
            raise NotImplementedError(f"No encoding implemented for {encoding}")
    # handle string dicts:
    if isinstance(data, str):
        data = json.loads(data)
    # writing the file for .json.gz
    if filename.endswith(".json.gz"):
        json_str = json.dumps(data, cls=encoding) + "\n"
        with gzip.open(filename, "w") as file:
            json_bytes = json_str.encode("utf-8")
            file.write(json_bytes)
    elif filename.endswith(".json"):
        indent = 4 if pretty else None
        try:
            with open(filename, "w") as file:
                json.dump(data, file, indent=indent)
        except TypeError as e:  # this must be improved
            raise TypeError(f"use .json.gz to encode numpy ndarray's serialization. {e}")
    else:
        raise IOError("File format must be .gz or .json.gz")


def get_all_pfam_ids():
    """Get all pfam ids from the pfam database."""
    table = get_root(__file__, retrace=2) / "data" / "pfam" / "pfam_families.tsv"
    data = pd.read_csv(table, sep="\t")
    print(data)
