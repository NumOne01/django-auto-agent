from django.conf import settings
from django.db import models


class AgentThread(models.Model):
    """Sidebar/permission anchor. Thread rows live in langgraph-db."""

    class Meta:
        managed = False
        default_permissions = ("view",)
        verbose_name = "Agent thread"
        verbose_name_plural = "Agent threads"


class AgentMemory(models.Model):
    """Sidebar/permission anchor. Memory rows live in the LangGraph store."""

    class Meta:
        managed = False
        default_permissions = ("view",)
        verbose_name = "Agent memory"
        verbose_name_plural = "Agent memories"


class AgentPrompt(models.Model):
    """Sidebar/permission anchor. Prompt overlays live in the LangGraph store."""

    class Meta:
        managed = False
        default_permissions = ("view",)
        verbose_name = "Agent prompt overlay"
        verbose_name_plural = "Agent prompt overlays"


class AgentMessage(models.Model):
    """User-visible chat turn. Independent of compacted LangGraph checkpoints."""

    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_CHOICES = (
        (ROLE_USER, "User"),
        (ROLE_ASSISTANT, "Assistant"),
    )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="agent_messages",
    )
    thread_id = models.CharField(max_length=128, db_index=True)
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    external_id = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]
        verbose_name = "Agent message"
        verbose_name_plural = "Agent messages"
        indexes = [
            models.Index(fields=["user", "thread_id", "created_at"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["thread_id", "external_id"],
                name="ai_agent_message_thread_external_id",
            ),
        ]

    def __str__(self):
        return f"{self.thread_id} {self.role}"
