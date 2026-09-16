from django.conf import settings
from django.db import models


class Item(models.Model):
    STATUS_OPEN = "OPEN"
    STATUS_CLOSED = "CLOSED"
    STATUS_CHOICES = (
        (STATUS_OPEN, "Open"),
        (STATUS_CLOSED, "Closed"),
    )

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="dummy_items",
    )
    name = models.CharField(max_length=64)
    secret = models.CharField(max_length=64, blank=True, default="")
    status = models.CharField(
        max_length=16, choices=STATUS_CHOICES, default=STATUS_OPEN
    )
    quantity = models.DecimalField(max_digits=12, decimal_places=4, default=0)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return self.name
