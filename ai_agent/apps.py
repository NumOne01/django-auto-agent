from django.apps import AppConfig


class AiAgentConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_agent"
    verbose_name = "AI Agent"
    agent_expose = False
    agent_memory_profile = "ai_agent.memory.supervisor_schemas.SupervisorProfile"
    agent_memory_collections = (
        "ai_agent.memory.supervisor_schemas.SupervisorFact",
        "ai_agent.memory.supervisor_schemas.DomainHabit",
    )
    agent_memory_episode = "ai_agent.memory.supervisor_schemas.SupervisorEpisode"
