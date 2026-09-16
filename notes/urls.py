from django.urls import path

from notes import views

urlpatterns = [
    path("public/", views.public_note, name="public_note"),
    path("private/", views.private_note, name="private_note"),
]
