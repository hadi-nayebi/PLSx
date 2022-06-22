import xml.etree.ElementTree as ETree
from pathlib import Path
from typing import Union

import pandas as pd
from Bio import SeqIO

from PLSx.utils.file_manager import get_root

UNIPROT_PREFIX = "{http://uniprot.org/uniprot}"
TARGET_DOMAINS = [
    # InterPro
    "IPR001721",
    "IPR002912",
    "IPR014864",
    "IPR018449",
    "IPR018717",
    "IPR019455",
    "IPR027795",
    "IPR039557",
    "IPR044074",
    "IPR014160",
    "IPR016540",
    "IPR022986",
    "IPR022988",
    "IPR023860",
    "IPR026249",
    "IPR027271",
    "IPR038110",
    "IPR044561",
    # CDD
    "cd04872",
    # PROSITE
    "PS51671",
    "PS51672",
    # SUPFAM
    # Pfam
    "PF13840",
]

Skip_list = [
    "bact",
    "lact",
    "activ",
    "bact",
    "fact",
    "charact",
    "actfrase",
    "uact",
    "interact",
    "impact",
    "vir_act",
    "ggcat",
    "pfl_act_enz",
    "actrans",
    "wall-act",
    "actyl",
    "actrfase",
]


def clean_tag(tag: str, prefix: str = None) -> str:
    """Clean tag."""
    if prefix is None:
        prefix = UNIPROT_PREFIX
    return tag.replace(prefix, "")


class DataPrep:

    root = get_root(__file__, retrace=2)

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
            all_act = []
            for item in root.iter(f"{UNIPROT_PREFIX}entry"):
                data = {}
                acc = item.find(UNIPROT_PREFIX + "accession").text
                # data["db_refs"] = [
                #     # (val.attrib.get("type"), val.attrib.get("id"))
                #     val
                #     for val in item.iter(UNIPROT_PREFIX + "dbReference")
                #     # if val.attrib.get("id") in TARGET_DOMAINS
                # ]
                # if len(data["db_refs"]) == 0:
                #     continue
                # else:
                #     for t, i in data["db_refs"]:
                #         if
                #         all_act.append(())

                for val in item.iter(UNIPROT_PREFIX + "dbReference"):
                    for prop in val.iter(UNIPROT_PREFIX + "property"):
                        value = prop.attrib.get("value").lower()
                        if "act" in value:
                            if not any([pattern in value for pattern in Skip_list]):
                                all_act.append(
                                    (
                                        val.attrib.get("type"),
                                        val.attrib.get("id"),
                                        prop.attrib.get("value"),
                                    )
                                )
                data["lineage"] = [
                    val.text
                    for val in item.find(UNIPROT_PREFIX + "organism")
                    .find(UNIPROT_PREFIX + "lineage")
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
                self.source[acc] = data
            print(all_act)
            print(len(all_act))

            # store_items = []
            # all_items = []
            # for item in prstree.findall("entry"):
            #     print(item)
            #     break

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
