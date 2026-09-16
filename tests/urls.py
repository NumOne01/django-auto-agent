from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/agent/", include("ai_agent.urls")),
    path("api/dummy/", include("dummy.urls")),
    path("api/notes/", include("notes.urls")),
]
