"""Admin for LangGraph long-term memory."""

from django.contrib import admin, messages
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import path, reverse
from django.utils.http import urlencode

from ai_agent.memory.namespaces import SUPERVISOR_LAYER
from ai_agent.models import AgentMemory

from ._helpers import page_offset, paginate, safe_next, user_change_url, user_display_name
from ._query import (
    delete_memories,
    list_memories,
    list_prompt_overlays,
    memories_grouped_for_user,
    resolve_user_query,
)

User = get_user_model()


@admin.register(AgentMemory)
class AgentMemoryAdmin(admin.ModelAdmin):
    change_list_template = "admin/ai_agent/memory_list.html"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return self.has_view_permission(request, obj)

    def get_queryset(self, request):
        return self.model.objects.none()

    def delete_view(self, request, object_id, extra_context=None):
        raise Http404("Use the memory list to delete selected memories.")

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path(
                "user/<int:user_id>/",
                self.admin_site.admin_view(self.user_view),
                name="ai_agent_agentmemory_user",
            ),
        ]
        return custom + urls

    def changelist_view(self, request, extra_context=None):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        list_url = reverse("admin:ai_agent_agentmemory_changelist")
        if request.method == "POST":
            return self._handle_delete_post(request, list_url)

        per_page = 25
        search = (request.GET.get("q") or "").strip()
        user_id = resolve_user_query(request.GET.get("user") or "")
        layer = (request.GET.get("layer") or "").strip()
        kind = (request.GET.get("kind") or "").strip()
        page_result = list_memories(
            search=search,
            user_id=user_id,
            layer=layer,
            kind=kind,
            limit=per_page,
            offset=page_offset(request, per_page=per_page),
        )
        page_obj, paginator = paginate(
            request, page_result.items, page_result.total, per_page=per_page
        )
        context = {
            **self.admin_site.each_context(request),
            "title": "Agent memories",
            "opts": self.model._meta,
            "page_obj": page_obj,
            "paginator": paginator,
            "memories": page_result.items,
            "error": page_result.error,
            "q": search,
            "user_filter": request.GET.get("user") or "",
            "layer_filter": layer,
            "kind_filter": kind,
            "supervisor_layer": SUPERVISOR_LAYER,
            "query_string": _query_string(request, drop=()),
            "can_delete": self.has_delete_permission(request),
        }
        return render(request, self.change_list_template, context)

    def user_view(self, request, user_id):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        user = get_object_or_404(User, pk=user_id)
        user_url = reverse("admin:ai_agent_agentmemory_user", args=[user.pk])
        if request.method == "POST":
            return self._handle_delete_post(request, user_url)
        grouped, error = memories_grouped_for_user(str(user.pk))
        if error and not grouped:
            grouped = {}
        user_overlays = list_prompt_overlays(
            user_id=str(user.pk), scope="local", limit=100, offset=0
        )
        global_overlays = list_prompt_overlays(scope="global", limit=50, offset=0)
        overlay_error = user_overlays.error or global_overlays.error
        context = {
            **self.admin_site.each_context(request),
            "title": f"Memories for {user_display_name(user)}",
            "opts": self.model._meta,
            "memory_user": user,
            "user_change_url": user_change_url(user),
            "grouped": grouped,
            "user_overlays": user_overlays.items,
            "global_overlays": global_overlays.items,
            "error": error or overlay_error,
            "supervisor_layer": SUPERVISOR_LAYER,
            "can_delete": self.has_delete_permission(request),
        }
        return render(request, "admin/ai_agent/memory_user.html", context)

    def _handle_delete_post(self, request, fallback_url):
        if not self.has_delete_permission(request):
            raise PermissionDenied
        selected = [item for item in request.POST.getlist("selected") if item.strip()]
        next_url = safe_next(request.POST.get("next"), fallback_url)
        if request.POST.get("action") != "delete":
            return redirect(next_url)
        if not selected:
            messages.error(request, "Select at least one memory.")
            return redirect(next_url)
        if request.POST.get("confirm") != "1":
            context = {
                **self.admin_site.each_context(request),
                "title": "Delete selected memories",
                "opts": self.model._meta,
                "selected": selected,
                "next": request.POST.get("next") or "",
                "noun": "memory",
                "noun_plural": "memories",
                "cancel_url": next_url,
            }
            return render(request, "admin/ai_agent/confirm_delete.html", context)
        deleted, error = delete_memories(selected)
        if error:
            messages.error(request, error)
        else:
            messages.success(
                request,
                f"Deleted {deleted} memor{'y' if deleted == 1 else 'ies'}.",
            )
        return redirect(next_url)

    def change_view(self, request, object_id, form_url="", extra_context=None):
        raise Http404("Memory items are listed on the changelist and per-user pages.")


def _query_string(request, *, drop=()):
    params = request.GET.copy()
    for key in drop:
        params.pop(key, None)
    return urlencode(params, doseq=True)
