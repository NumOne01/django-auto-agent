"""Shared presentation helpers for AI agent admin views."""

from __future__ import annotations

from django.core.paginator import EmptyPage, Paginator
from django.urls import NoReverseMatch, reverse


def user_display_name(user) -> str:
    """Prefer first/last name, then ``get_username()``."""
    if user is None:
        return ""
    name = " ".join(
        part
        for part in (getattr(user, "first_name", None), getattr(user, "last_name", None))
        if part
    )
    if name:
        return name
    getter = getattr(user, "get_username", None)
    if callable(getter):
        return str(getter() or "").strip()
    return ""


def user_change_url(user) -> str:
    """Django admin change URL for the configured user model."""
    if user is None:
        return ""
    opts = user._meta
    try:
        return reverse(
            f"admin:{opts.app_label}_{opts.model_name}_change",
            args=[user.pk],
        )
    except NoReverseMatch:
        return ""


def paginate(request, items, total: int, *, per_page: int = 25):
    """Build a Django page from an already-sliced list plus a known total."""
    paginator = Paginator(range(total), per_page)
    page_number = request.GET.get("p") or 1
    try:
        page = paginator.page(page_number)
    except EmptyPage:
        page = paginator.page(paginator.num_pages or 1)
    page.object_list = items
    return page, paginator


def page_offset(request, *, per_page: int = 25) -> int:
    try:
        page_number = int(request.GET.get("p") or 1)
    except (TypeError, ValueError):
        page_number = 1
    if page_number < 1:
        page_number = 1
    return (page_number - 1) * per_page


def safe_next(raw: str, fallback: str) -> str:
    """Allow only a relative query string on the same admin path."""
    value = (raw or "").strip()
    path, _, _ = fallback.partition("?")
    if value.startswith("?") and "://" not in value and "\n" not in value:
        return path + value
    return fallback
