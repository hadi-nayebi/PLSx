from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

from .models import Protein

# def index(request):
#     protein_list = Protein.objects.order_by("uniprot_id")
#     template = loader.get_template("dataView/index.html")
#     context = {"protein_list": protein_list}
#     return HttpResponse(template.render(context, request))


def index(request):
    protein_list = Protein.objects.order_by("uniprot_id")
    context = {"protein_list": protein_list}
    return render(request, "dataView/index.html", context)


def protein(request, uniprot_id):
    return HttpResponse("Protein View: {}".format(uniprot_id))
