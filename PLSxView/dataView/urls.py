from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("protein/<str:uniprot_id>", views.protein, name="protein"),
]
