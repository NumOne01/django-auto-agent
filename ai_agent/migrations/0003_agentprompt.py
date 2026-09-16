from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_agent", "0002_agentmessage"),
    ]

    operations = [
        migrations.CreateModel(
            name="AgentPrompt",
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
            ],
            options={
                "verbose_name": "Agent prompt overlay",
                "verbose_name_plural": "Agent prompt overlays",
                "managed": False,
                "default_permissions": ("view",),
            },
        ),
    ]
