import pickle
from os import system

from numpy import array, int8, zeros

from Plsx.protein.exceptions import ImmutablePropertyError
from Plsx.protein.utils import (
    AA_PADDING_NDX,
    SS_PADDING_NDX,
    get_tax_id_from_uniprot_metadata,
    is_array_int,
    is_array_p,
    is_protein_sequence,
    is_ss_sequence,
    ndx_to_seq,
    seq_to_ndx,
)
from Plsx.utils.base_logger import base_logger
from Plsx.utils.data_pipeline_uniprot import DataPipeline
from Plsx.utils.file_manager import get_root


class Protein:

    root = get_root(__file__, retrace=2)

    def __init__(self, name: str) -> None:
        """Initialize the Protein."""
        self._name = name

        # proteins dir if does not exist, create it
        proteins_dir = self.root / "data" / "proteins"
        if not proteins_dir.exists():
            proteins_dir.mkdir(parents=True)

        # protein dir
        self._path = self.root / "data" / "proteins" / f"{name}.pkl.bz2"

        # Protein attrs
        self._seq_ndx = None  # seq index used for training
        self._padding = None
        self._checksum = None
        self._seq_ss = None  # ss index used for training
        self._annotations = {}  # annotations used for training
        self._dataset = {
            "name": "",
            "mode": "",
        }  # dataset attrs used for the training purposes
        self.metadata = {}
        # add name to metadata
        self.add_metadata("names", {name})

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

    @property
    def seq_ndx(self):
        return self._seq_ndx

    @property
    def checksum(self):
        return self._checksum

    @property
    def seq_ss(self):
        return self._seq_ss

    @property
    def annotations(self):
        return self._annotations

    @property
    def dataset(self):
        return self._dataset

    @property
    def padding(self):
        return self._padding

    @seq_ndx.setter
    def seq_ndx(self, seq):
        try:
            if self._checksum is not None:
                raise ImmutablePropertyError(attr="Protein sequence")
            if is_protein_sequence(seq):
                self._seq_ndx = seq_to_ndx(seq, keys="aa_keys")
            elif is_array_int(seq, max_val=AA_PADDING_NDX):
                self._seq_ndx = array(seq, dtype=int8)
            self._padding = len(self._seq_ndx[self._seq_ndx == AA_PADDING_NDX]) // 2
        except ImmutablePropertyError:
            base_logger.logger.info(f"Request to update {self.name} sequence failed.")

    @checksum.setter
    def checksum(self, value):
        try:
            if self._checksum is not None:
                if value == self._checksum:
                    return
                raise ImmutablePropertyError(attr="Checksum")
            self._checksum = value
        except ImmutablePropertyError:
            base_logger.logger.info(f"Request to update {self.name} checksum failed.")

    @seq_ss.setter
    def seq_ss(self, seq):
        # TODO cases were ss_seq is entered with no padding?
        if is_ss_sequence(seq):
            assert len(seq) == len(self._seq_ndx)
            self._seq_ss = seq_to_ndx(seq, keys="ss_keys")
        elif is_array_int(seq, max_val=SS_PADDING_NDX):
            assert len(seq) == len(self._seq_ndx)
            self._seq_ss = array(seq, dtype=int8)

    @annotations.setter
    def annotations(self, value):
        # TODO cases were annotations are entered with no padding?
        if isinstance(value, dict):
            for key, item in value.items():
                if is_array_p(item):
                    assert len(item) == len(self._seq_ndx)
                    self._annotations[key] = array(item, dtype=float)

    @dataset.setter
    def dataset(self, value):
        if isinstance(value, dict):
            assert "name" in value.keys()
            assert "mode" in value.keys()
            self._dataset["name"] = value["name"]
            self._dataset["mode"] = value["mode"]

    def rename(self, new_name, save=False):
        if new_name != self.name:
            if save:
                self.remove_file()
            self._name = new_name
            self._path = self.root / "data" / "proteins" / f"{self.name}.pkl.bz2"
            self.add_metadata("names", {new_name})
            if save:
                self.save_file(overwrite=True)

    def remove_file(self):
        if self._path.exists():
            system(f"rm {self._path}")

    def save_file(self, overwrite=False):
        if not overwrite:
            i = 0
            original_name = self._name[:]
            while self._path.exists():
                i += 1
                self.rename(f"{original_name}_{i}")
        with open(self._path, "wb") as f:
            pickle.dump(self, f)

    def add_metadata(self, key, value):
        if key in self.metadata.keys():
            if isinstance(self.metadata[key], list):
                if isinstance(value, list):
                    for item in value:
                        self.metadata[key].append(item)
                else:
                    self.metadata[key].append(value)
            if isinstance(self.metadata[key], set):
                if isinstance(value, set):
                    for item in value:
                        self.metadata[key].add(item)
                else:
                    self.metadata[key].add(value)
        else:
            self.metadata[key] = value

    def del_metadata(self, key):
        del self.metadata[key]

    def aa_seq(self, no_padding=True):
        if no_padding:
            return ndx_to_seq(self._seq_ndx, keys="aa_keys").replace("*", "")
        return ndx_to_seq(self._seq_ndx)

    def ss_seq(self, no_padding=True):
        if no_padding:
            return ndx_to_seq(self._seq_ss, keys="ss_keys").replace("*", "")
        return ndx_to_seq(self._seq_ss)

    def size(self, padding_ndx=20):
        return len(self._seq_ndx[self._seq_ndx != padding_ndx])

    def __str__(self):
        output = f"name: {self.name}\n"
        output += f"seq: {self.aa_seq()}\n"
        output += f"checksum: {self._checksum}\n"
        output += f"dataset: {self.dataset}\n"
        has_ss = "has_ss" if self._seq_ss is not None else "no_ss"
        has_padding = f"padded by {self._padding}" if self._padding > 0 else ""
        output += f"attrs: {has_ss} | {has_padding}\n"
        output += f"annotations: {', '.join(self._annotations.keys()) if len(self._annotations) > 0 else ''}\n"
        output += (
            f"metadata keys: {', '.join(self.metadata.keys()) if len(self.metadata) > 0 else ''}\n"
        )
        return output

    def get_data(self, ss=False, annotations=None):
        data = {"ndx": self._seq_ndx}
        if ss and self._seq_ss is not None:
            data["ss"] = self._seq_ss
        if annotations is not None and isinstance(annotations, list):
            for key, value in self._annotations.items():
                if key in annotations:
                    data[key] = value
        return data

    def update_uniprot_metadata(self, db_ref_limit=20, data_pipeline=None):
        if data_pipeline is None:
            data_pipeline = DataPipeline()
        aa_seq = self.aa_seq()
        result = data_pipeline.fetch_by_seq(aa_seq)
        if result == {}:
            self.add_metadata("errors", {"uniprot returns {}."})
            return
            # parse and add uniprot metadata to metadata
        if "sequence" in result.keys():
            if aa_seq != result["sequence"]["content"]:
                self.add_metadata("errors", {"seq do nat match uniprot seq."})
                # base_logger.logger.info(f"sequence for {self.name} does not match uniprot results")
                return
            self.checksum = result["sequence"]["checksum"]
        else:
            self.add_metadata("errors", {"uniprot metadata missing sequence."})
        # update name
        self.rename(result["accession"])
        # store signatureSequenceMatch
        if "signatureSequenceMatch" in result.keys():
            self.add_metadata("UP_signatureSequenceMatch", result["signatureSequenceMatch"])
        else:
            self.add_metadata("errors", {"uniprot metadata missing signatureSequenceMatch."})
        # store dbReference
        if "dbReference" in result.keys():
            # get tax id
            self.add_metadata("NCBI_taxonomy_id", get_tax_id_from_uniprot_metadata(result))
            state = "complete" if len(result["dbReference"]) < db_ref_limit else "partial"
            up_db_reference = {
                "value": result["dbReference"][:db_ref_limit],
                "state": state,
            }
            self.add_metadata("UP_dbReference", up_db_reference)
        else:
            self.add_metadata("errors", {"uniprot metadata missing dbReference."})

    def copy(self, other):
        assert isinstance(other, Protein)
        for key, value in other.__dict__.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = value

    def add_annotation(self, name, database_domain_ids):
        assert "UP_signatureSequenceMatch" in self.metadata.keys()
        has = False
        vals = zeros(self._seq_ndx.shape, dtype=float)
        for item in self.metadata["UP_signatureSequenceMatch"]:
            for database, domain_ids in database_domain_ids.items():
                if item["database"] == database and item["id"] in domain_ids:
                    for pos in item["lcn"]:
                        start = self.padding + pos["start"]
                        end = self.padding + pos["end"]
                        vals[start:end] += 1
                        has = True
        if has:
            vals = vals / max(vals)
            self.annotations = {name: vals}

    # def get_ss_pattern(self, ndx):
    #     if "structure_class" in self.metadata.keys():
    #         ss_pattern, _ = ss_pattern_dict[self.metadata["structure_class"][0]]
    #         return ss_pattern[ndx]

    # def get_pfam(self, ndx):
    #     if "structure_class" in self.metadata.keys():
    #         _, pfam = ss_pattern_dict[self.metadata["structure_class"][0]]
    #         return pfam[ndx]
