from django.urls import path

from ai_agent import views

urlpatterns = [
    path("threads/", views.list_agent_threads, name="agent_threads"),
    path(
        "threads/<str:thread_id>/messages/",
        views.list_agent_thread_messages,
        name="agent_thread_messages",
    ),
    path("memories/", views.list_agent_memories, name="agent_memories"),
    path(
        "memories/<str:layer>/<str:kind>/<path:key>/",
        views.agent_memory_detail,
        name="agent_memory_detail",
    ),
]
