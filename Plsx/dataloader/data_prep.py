import xml.etree.ElementTree as ETree
from pathlib import Path
from typing import Union

import pandas as pd
from Bio import SeqIO

from Plsx.dataloader.utils import read_json
from Plsx.utils.file_manager import get_root


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

    def load_source(self, path: Union[Path, str]) -> None:
        """Load source data."""
        if isinstance(path, str):
            path = Path(path)
        assert path.exists(), f"{path} does not exist"
        self.source_path = path
        self.source = {}
        if ".xml" in path.suffixes:
            print("Loading XML file...")
            prstree = ETree.parse(path)
            print("Parsing XML file...")
            root = prstree.getroot()
            all_act = set()
            for item in root.iter(f"{self.config['prefix']}entry"):
                data = {}
                acc = item.find(self.config["prefix"] + "accession").text
                # data["db_refs"] = [
                #     # (val.attrib.get("type"), val.attrib.get("id"))
                #     val
                #     for val in item.iter(self.config['prefix'] + "dbReference")
                #     # if val.attrib.get("id") in self.config["target_domains"]
                # ]
                # if len(data["db_refs"]) == 0:
                #     continue
                # else:
                #     for t, i in data["db_refs"]:
                #         if
                #         all_act.append(())

                for val in item.iter(self.config["prefix"] + "dbReference"):
                    if not db in self.config["skip_db"]:
                        for prop in val.iter(self.config["prefix"] + "property"):
                            value = prop.attrib.get("value").lower()
                            db = val.attrib.get("type")
                            if "act" in value:
                                if not any(
                                    [pattern in value for pattern in self.config["skip_ids"]]
                                ):
                                    all_act.add(
                                        (
                                            val.attrib.get("type"),
                                            val.attrib.get("id"),
                                            prop.attrib.get("value"),
                                        )
                                    )
                data["lineage"] = [
                    val.text
                    for val in item.find(self.config["prefix"] + "organism")
                    .find(self.config["prefix"] + "lineage")
                    .iter()
                    if "\n" not in val.text
                ]

                # for metadata in item:
                #     tag = clean_tag(metadata.tag)
                #     if tag == "accession":
                #         acc = metadata.text
                #     elif tag == "organism":
                #         for c in metadata:
                #             if clean_tag(c.tag) == "lineage":
                #                 data["lineage"] = [
                #                     val.text for val in c.iter() if "\n" not in val.text
                #                 ]
                #     elif tag == "dbReference":
                #
                #         for c in metadata:
                #             if clean_tag(c.tag) == "property":
                #                 data[clean_tag(c.tag)] = c.text
                # print(acc)
                # break
                # self.source[acc] = data
            print(all_act)
            print(len(all_act))

            # store_items = []
            # all_items = []
            # for item in prstree.findall("entry"):
            #     print(item)
            #     break
        del prstree
        del root

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
