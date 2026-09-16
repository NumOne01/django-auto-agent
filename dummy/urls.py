from django.urls import path

from dummy import views

urlpatterns = [
    path("items/", views.list_items, name="dummy_item_list"),
    path("items/create/", views.create_item, name="dummy_item_create"),
    path("items/<int:pk>/", views.item_detail, name="dummy_item_detail"),
    path("hidden/", views.dummy_hidden, name="dummy_hidden"),
    path("webhooks/payment/", views.dummy_webhook, name="dummy_webhook"),
    path("internal/hook/", views.dummy_internal, name="dummy_internal"),
    path("upload/", views.dummy_upload, name="dummy_upload"),
    path("tickets/", views.dummy_create_ticket, name="dummy_create_ticket"),
    path("receipt/", views.dummy_receipt, name="dummy_receipt"),
]
