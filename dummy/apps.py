from django.apps import AppConfig


class DummyConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "dummy"
    verbose_name = "Dummy Shop"
    agent_expose = True
    agent_description = "List and create catalog items for the authenticated user."
    agent_exclude = ("dummy_hidden",)
    agent_memory_profile = "dummy.memory_schemas.DummyProfile"
    agent_memory_collections = (
        "ai_agent.memory.schemas.SemanticFact",
        "dummy.memory_schemas.DummyAccount",
    )
    agent_memory_episode = "ai_agent.memory.schemas.Episode"
