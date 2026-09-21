"""Django connection hygiene for Agent Server threads.

LangGraph never fires ``request_started`` / ``request_finished``, so
``close_old_connections()`` must run on the same thread as the ORM
(``sync_to_async(..., thread_sensitive=True)``). Calling it on the event
loop does not see the thread-local connection (Django ticket #35583).
"""

from __future__ import annotations


def refresh_db_connections():
    """Drop stale thread-local connections. Skip Django tests (SQLite TestCase)."""
    from django.conf import settings
    from django.db import close_old_connections

    if getattr(settings, "TESTING", False):
        return
    close_old_connections()
