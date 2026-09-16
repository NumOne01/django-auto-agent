"""Admin for LangGraph conversation threads."""

from django.contrib import admin, messages
from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.http import urlencode

from ai_agent.models import AgentThread

from ._helpers import page_offset, paginate, safe_next
from ._query import delete_threads, get_thread, list_threads, resolve_user_query


@admin.register(AgentThread)
class AgentThreadAdmin(admin.ModelAdmin):
    change_list_template = "admin/ai_agent/thread_list.html"
    change_form_template = "admin/ai_agent/thread_detail.html"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return self.has_view_permission(request, obj)

    def get_queryset(self, request):
        return self.model.objects.none()

    def delete_view(self, request, object_id, extra_context=None):
        raise Http404("Use the thread list to delete selected threads.")

    def changelist_view(self, request, extra_context=None):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        list_url = reverse("admin:ai_agent_agentthread_changelist")
        if request.method == "POST":
            return self._handle_delete_post(request, list_url)

        per_page = 25
        search = (request.GET.get("q") or "").strip()
        owner_id = resolve_user_query(request.GET.get("user") or "")
        include_children = request.GET.get("include_children") == "1"
        page_result = list_threads(
            search=search,
            owner_id=owner_id,
            include_children=include_children,
            limit=per_page,
            offset=page_offset(request, per_page=per_page),
        )
        page_obj, paginator = paginate(
            request, page_result.items, page_result.total, per_page=per_page
        )
        context = {
            **self.admin_site.each_context(request),
            "title": "Agent threads",
            "opts": self.model._meta,
            "cl": self,
            "page_obj": page_obj,
            "paginator": paginator,
            "threads": page_result.items,
            "error": page_result.error,
            "q": search,
            "user_filter": request.GET.get("user") or "",
            "include_children": include_children,
            "query_string": _query_string(request, drop=()),
            "can_delete": self.has_delete_permission(request),
        }
        return render(request, self.change_list_template, context)

    def _handle_delete_post(self, request, list_url):
        if not self.has_delete_permission(request):
            raise PermissionDenied
        selected = [item for item in request.POST.getlist("selected") if item.strip()]
        next_url = safe_next(request.POST.get("next"), list_url)
        if request.POST.get("action") != "delete":
            return redirect(next_url)
        if not selected:
            messages.error(request, "Select at least one thread.")
            return redirect(next_url)
        if request.POST.get("confirm") != "1":
            context = {
                **self.admin_site.each_context(request),
                "title": "Delete selected threads",
                "opts": self.model._meta,
                "selected": selected,
                "next": request.POST.get("next") or "",
                "noun": "thread",
                "noun_plural": "threads",
                "cancel_url": next_url,
            }
            return render(request, "admin/ai_agent/confirm_delete.html", context)
        deleted, error = delete_threads(selected)
        if error:
            messages.error(request, error)
        else:
            messages.success(
                request, f"Deleted {deleted} thread{'s' if deleted != 1 else ''}."
            )
        return redirect(next_url)

    def change_view(self, request, object_id, form_url="", extra_context=None):
        return self.detail_view(request, object_id)

    def detail_view(self, request, object_id):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        thread, error = get_thread(str(object_id))
        if error is None and thread is None:
            raise Http404("Thread not found")
        context = {
            **self.admin_site.each_context(request),
            "title": f"Thread {object_id}",
            "opts": self.model._meta,
            "thread": thread,
            "error": error,
            "original": thread,
        }
        return render(request, self.change_form_template, context)


def _query_string(request, *, drop=()):
    params = request.GET.copy()
    for key in drop:
        params.pop(key, None)
    return urlencode(params, doseq=True)
