#!/usr/bin/env python
# coding: utf-8

import time
import xml.etree.ElementTree as ETree
from asyncore import write
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from Plsx.dataloader.utils import XML_Obj, append_id, read_json, write_json
from Plsx.protein.protein import Protein
from Plsx.utils.file_manager import get_root
from Plsx.utils.parallel_jobs import asyncfunc, limit


def clean_tag(tag: str, prefix: str) -> str:
    """Clean tag."""
    return tag.replace(prefix, "")


class DataPrep:

    root = get_root(__file__, retrace=2)
    config = read_json(root / "config" / "prep_data.json")

    def __init__(self):
        """Initialize the DataPrep."""
        self.source_path = None
        self.source = None
        self.stat = {}  # stat dict for DataPrep

    def skip_db(self, db: str) -> bool:
        """Skip db."""
        return db in self.config["skip_db"]

    def skip_ids(self, value: str) -> bool:
        """Skip ids."""
        return any([pattern in value for pattern in self.config["skip_ids"]])

    def is_target_domain(self, db: str, domain: str) -> bool:
        """Is target domain."""
        if db not in self.config["target_domains"].keys():
            return False
        return domain in self.config["target_domains"][db]

    def is_target_domain_by_pr(self, pr: Protein) -> bool:
        """Is target domain by pr."""
        if "UP_signatureSequenceMatch" not in pr.metadata.keys():
            return False
        for item in pr.metadata["UP_signatureSequenceMatch"]:
            domain_db = item["database"]
            domain_id = item["id"]
            if self.is_target_domain(domain_db, domain_id):
                return True
        return False

    @limit(number=10)
    @asyncfunc
    def add_to_dataset(self, item: ETree.Element) -> None:
        """Add to dataset."""
        time.sleep(np.random.uniform() * 60)
        seq = item.find(f"{self.config['prefix']}sequence").text
        acc = item.find(f"{self.config['prefix']}accession").text
        pr = Protein(acc)
        pr.seq_ndx = seq
        pr.update_uniprot_metadata()
        if self.is_target_domain_by_pr(pr):
            pr.save_file()
            self.stat[pr.checksum] = pr.name

    def load_source(self, path: Union[Path, str]) -> None:
        """Load source data."""
        if isinstance(path, str):
            path = Path(path)
        assert path.exists(), f"{path} does not exist"
        self.source = {}
        if ".xml" in path.suffixes:
            print("Loading XML file...")
            prstree = ETree.parse(path)
            print("Parsing XML file...")
            root = prstree.getroot()
            for item in tqdm(root.iter(f"{self.config['prefix']}entry")):
                self.add_to_dataset(item)
                if len(self.stat) % 1000 == 0:
                    write_json(self.stat, self.root / "data" / "proteins" / "stat.json")
                    time.sleep(10)

    # def load_source(self, path: Union[Path, str]) -> None:
    #     """Load source data."""
    #     if isinstance(path, str):
    #         path = Path(path)
    #     assert path.exists(), f"{path} does not exist"
    #     if self.config["save_as"] is not None:
    #         self.source_path = append_id(path, self.config["save_as"])
    #     self.source = XML_Obj(self.config["save_as"])
    #     if ".xml" in path.suffixes:
    #         print("Loading XML file...")
    #         prstree = ETree.parse(path)
    #         print("Parsing XML file...")
    #         root = prstree.getroot()
    #         # all_act = set() # used for filtering against keywords
    #         for item in tqdm(root.iter(f"{self.config['prefix']}entry")):
    #             for val in item.iter(self.config["prefix"] + "dbReference"):
    #                 # used for filtering against keywords
    #                 # if self.skip_db(val.attrib.get("type")):
    #                 #     continue
    #                 # for prop in val.iter(self.config["prefix"] + "property"):
    #                 #     value = prop.attrib.get("value").lower()
    #                 #     if "act" not in value or self.skip_ids(value):
    #                 #         continue
    #                 #     all_act.add(
    #                 #         (
    #                 #             val.attrib.get("type"),
    #                 #             val.attrib.get("id"),
    #                 #             prop.attrib.get("value"),
    #                 #         )
    #                 #     )
    #                 # filter by target domains
    #                 if not self.is_target_domain(val.attrib.get("type"), val.attrib.get("id")):
    #                     continue
    #                 self.source.add_child(item)
    #                 break

    #             # data["lineage"] = [
    #             #     val.text
    #             #     for val in item.find(self.config["prefix"] + "organism")
    #             #     .find(self.config["prefix"] + "lineage")
    #             #     .iter()
    #             #     if "\n" not in val.text
    #             # ]

    #             # for metadata in item:
    #             #     tag = clean_tag(metadata.tag)
    #             #     if tag == "accession":
    #             #         acc = metadata.text
    #             #     elif tag == "organism":
    #             #         for c in metadata:
    #             #             if clean_tag(c.tag) == "lineage":
    #             #                 data["lineage"] = [
    #             #                     val.text for val in c.iter() if "\n" not in val.text
    #             #                 ]
    #             #     elif tag == "dbReference":
    #             #
    #             #         for c in metadata:
    #             #             if clean_tag(c.tag) == "property":
    #             #                 data[clean_tag(c.tag)] = c.text
    #             # print(acc)
    #             # break
    #             # self.source[acc] = data
    #         # print(all_act)
    #         # print(len(all_act))
    #         # saving filtered xml
    #         self.source.save(self.source_path)
    #         # store_items = []
    #         # all_items = []
    #         # for item in prstree.findall("entry"):
    #         #     print(item)
    #         #     break
    #         del prstree
    #         del root

    def get_families(self, keyword: Union[str, list]) -> None:
        """Get families."""
        if isinstance(keyword, str):
            keyword = [keyword]
        if self.source_metadata is None:
            raise ValueError("source_metadata is None")
        # search for keyword in metadata
        # self.source_metadata = self.source_metadata.loc[
        #     self.source_metadata["organism"].str.contains(
        #         "|".join(keyword), case=False)
        # ]


if __name__ == "__main__":
    data_prep = DataPrep()
    source = data_prep.root / "data" / "uniprot" / "uniprot_sprot.xml"
    if not source.exists():
        raise ValueError(f"{source} does not exist")
    data_prep.load_source(source)
    exit()
