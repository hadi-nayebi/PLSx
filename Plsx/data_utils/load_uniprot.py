#!/usr/bin/env python
# coding: utf-8

from os.path import dirname

# an script to load uniprot data into a sqlite database, the data includes the protein sequences and signatureSequenceMatch metadata
from pathlib import Path

import pandas as pd
from Bio import Entrez

from Plsx.database.sqlitedb import SqliteDB

root = Path(dirname(__file__)).parent.parent
# define a local sqilte database
db = SqliteDB()
db.create("uniprot")

# list of species with full genome sequences, downloaded from NCBI
species = pd.read_csv(root / "data" / "genomes.csv")
# print(species["Size(Mb)"].sum())

Entrez.email = "nayebiga@msu.edu"
for name in species["#Organism Name"]:
    eSearch = Entrez.esearch(db="genome", term="name")
    res = Entrez.read(eSearch)
    for k in res:
        print(k, "=", res[k])
    break
