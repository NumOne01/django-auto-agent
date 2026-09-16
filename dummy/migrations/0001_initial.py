from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Item",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=64)),
                ("secret", models.CharField(blank=True, default="", max_length=64)),
                (
                    "status",
                    models.CharField(
                        choices=[("OPEN", "Open"), ("CLOSED", "Closed")],
                        default="OPEN",
                        max_length=16,
                    ),
                ),
                (
                    "quantity",
                    models.DecimalField(decimal_places=4, default=0, max_digits=12),
                ),
                (
                    "owner",
                    models.ForeignKey(
                        on_delete=models.CASCADE,
                        related_name="dummy_items",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={"ordering": ["id"]},
        ),
    ]
