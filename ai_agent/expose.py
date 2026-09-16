"""Decorators that mark DRF views for LangGraph tool exposure."""


def _set_agent_attr(view, name, value):
    setattr(view, name, value)
    cls = getattr(view, "cls", None)
    if cls is not None:
        setattr(cls, name, value)
    wrapped = getattr(view, "__wrapped__", None)
    if wrapped is not None and wrapped is not view:
        setattr(wrapped, name, value)
    return view


def get_view_attr(view, name, default=None):
    """Read an agent flag from a DRF wrapped view or its original function."""
    if hasattr(view, name):
        return getattr(view, name)
    cls = getattr(view, "cls", None)
    if cls is not None and hasattr(cls, name):
        return getattr(cls, name)
    wrapped = getattr(view, "__wrapped__", None)
    if wrapped is not None and wrapped is not view:
        return get_view_attr(wrapped, name, default)
    return default


def agent_expose(
    func=None, *, confirm=None, description=None, response_serializer=None
):
    """Mark a view as exposed to the AI agent.

    Can expose a single function even when the Django app is not opted in.
    ``confirm`` overrides the default mutation HITL flag for this view.
    ``response_serializer`` is an optional DRF serializer class that reshapes
    2xx ``response.data`` before it is returned as a tool result.
    """

    def apply(view):
        _set_agent_attr(view, "_agent_expose", True)
        if confirm is not None:
            _set_agent_attr(view, "_agent_confirm", confirm)
        if description is not None:
            _set_agent_attr(view, "_agent_description", description)
        if response_serializer is not None:
            _set_agent_attr(view, "_agent_response_serializer", response_serializer)
        return view

    if func is not None:
        return apply(func)
    return apply


def agent_exclude(func=None):
    """Exclude a view from the AI agent even when its app is opted in."""

    def apply(view):
        _set_agent_attr(view, "_agent_exclude", True)
        return view

    if func is not None:
        return apply(func)
    return apply
