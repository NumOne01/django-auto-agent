"""Admin for the user-visible agent transcript."""

from django.contrib import admin
from django.contrib.auth import get_user_model

from ai_agent.models import AgentMessage


@admin.register(AgentMessage)
class AgentMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "thread_id", "role", "preview", "created_at")
    list_filter = ("role", "created_at")
    readonly_fields = (
        "user",
        "thread_id",
        "role",
        "content",
        "external_id",
        "created_at",
    )
    ordering = ("-created_at", "-id")
    list_select_related = ("user",)

    def get_search_fields(self, request):
        field = get_user_model().USERNAME_FIELD
        return ("thread_id", "content", f"user__{field}")

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def preview(self, obj):
        text = " ".join((obj.content or "").split())
        if len(text) > 80:
            return text[:80].rstrip() + "…"
        return text

    preview.short_description = "Preview"
