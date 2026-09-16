"""Admin observability for local and global prompt overlays."""

from django.contrib import admin, messages
from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.http import urlencode

from ai_agent.memory.namespaces import SUPERVISOR_LAYER
from ai_agent.models import AgentPrompt

from ._helpers import page_offset, paginate, safe_next, user_change_url
from ._query import (
    delete_prompt_overlays,
    get_prompt_overlay,
    list_prompt_overlays,
    resolve_user_query,
)


@admin.register(AgentPrompt)
class AgentPromptAdmin(admin.ModelAdmin):
    change_list_template = "admin/ai_agent/prompt_list.html"
    change_form_template = "admin/ai_agent/prompt_detail.html"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return self.has_view_permission(request, obj)

    def get_queryset(self, request):
        return self.model.objects.none()

    def changelist_view(self, request, extra_context=None):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        list_url = reverse("admin:ai_agent_agentprompt_changelist")
        if request.method == "POST":
            return self._handle_delete_post(request, list_url)

        per_page = 25
        search = (request.GET.get("q") or "").strip()
        user_id = resolve_user_query(request.GET.get("user") or "")
        layer = (request.GET.get("layer") or "").strip()
        global_page = list_prompt_overlays(
            search=search,
            layer=layer,
            scope="global",
            limit=50,
            offset=0,
        )
        local_page = list_prompt_overlays(
            search=search,
            user_id=user_id,
            layer=layer,
            scope="local",
            limit=per_page,
            offset=page_offset(request, per_page=per_page),
        )
        page_obj, paginator = paginate(
            request, local_page.items, local_page.total, per_page=per_page
        )
        error = global_page.error or local_page.error
        context = {
            **self.admin_site.each_context(request),
            "title": "Agent prompt overlays",
            "opts": self.model._meta,
            "page_obj": page_obj,
            "paginator": paginator,
            "global_overlays": global_page.items,
            "local_overlays": local_page.items,
            "error": error,
            "q": search,
            "user_filter": request.GET.get("user") or "",
            "layer_filter": layer,
            "supervisor_layer": SUPERVISOR_LAYER,
            "query_string": _query_string(request, drop=()),
            "can_delete": self.has_delete_permission(request),
            "has_overlays": bool(global_page.items or local_page.items),
        }
        return render(request, self.change_list_template, context)

    def _handle_delete_post(self, request, fallback_url):
        if not self.has_delete_permission(request):
            raise PermissionDenied
        selected = [item for item in request.POST.getlist("selected") if item.strip()]
        next_url = safe_next(request.POST.get("next"), fallback_url)
        if request.POST.get("action") != "delete":
            return redirect(next_url)
        if not selected:
            messages.error(request, "Select at least one overlay.")
            return redirect(next_url)
        if request.POST.get("confirm") != "1":
            context = {
                **self.admin_site.each_context(request),
                "title": "Delete selected overlays",
                "opts": self.model._meta,
                "selected": selected,
                "next": request.POST.get("next") or "",
                "noun": "overlay",
                "noun_plural": "overlays",
                "cancel_url": next_url,
            }
            return render(request, "admin/ai_agent/confirm_delete.html", context)
        deleted, error = delete_prompt_overlays(selected)
        if error:
            messages.error(request, error)
        else:
            messages.success(
                request,
                f"Deleted {deleted} overlay{'s' if deleted != 1 else ''}.",
            )
        return redirect(next_url)

    def change_view(self, request, object_id, form_url="", extra_context=None):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        overlay, error = get_prompt_overlay(object_id)
        if overlay is None and not error:
            raise Http404("Prompt overlay not found.")
        context = {
            **self.admin_site.each_context(request),
            "title": _overlay_title(overlay),
            "opts": self.model._meta,
            "overlay": overlay,
            "error": error,
            "supervisor_layer": SUPERVISOR_LAYER,
            "user_change_url": user_change_url(getattr(overlay, "user", None)),
        }
        return render(request, self.change_form_template, context)

    def delete_view(self, request, object_id, extra_context=None):
        raise Http404("Use the overlay list to delete selected overlays.")


def _overlay_title(overlay) -> str:
    if overlay is None:
        return "Prompt overlay"
    layer = overlay.layer_display
    if overlay.scope == "global":
        return f"Global overlay · {layer}"
    return f"User addendum · {layer}"


def _query_string(request, *, drop=()):
    params = request.GET.copy()
    for key in drop:
        params.pop(key, None)
    return urlencode(params, doseq=True)
