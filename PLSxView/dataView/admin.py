from django.contrib import admin

from .models import Annotations, Protein

admin.site.register(Protein)
admin.site.register(Annotations)
