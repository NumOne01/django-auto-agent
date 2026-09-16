from django.apps import AppConfig


class NotesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "notes"
    verbose_name = "Notes"
    agent_expose = False
    agent_description = "Optional notes the host opted in per-view."
