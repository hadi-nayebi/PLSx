from django.db import models
from sqlalchemy import ForeignKey


class Protein(models.Model):
    """
    A class to represent a protein."""

    uniprot_id = models.CharField(max_length=20, unique=True, primary_key=True)
    sequence = models.TextField()
    secondary_structure = models.TextField(blank=True)

    def __str__(self):
        """Return the string representation of the protein."""
        return f"{self.uniprot_id} -> \nAAS: {self.sequence}\nSSS: {self.secondary_structure}"


class Annotations(models.Model):
    """
    A class to represent annotations."""

    uniprot_id = models.CharField(max_length=20)
    annotation = models.TextField()
    start = models.IntegerField()
    end = models.IntegerField()
