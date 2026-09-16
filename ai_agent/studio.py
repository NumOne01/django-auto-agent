"""LangGraph Studio / Agent Server entrypoint.

Boots Django so URLconf discovery and in-process DRF tools work, then exports
the compiled supervisor. Persistence is left to the Agent Server.

Run from the repo root. Set ``DJANGO_SETTINGS_MODULE`` or
``AI_AGENT_STUDIO_DJANGO_SETTINGS`` (compose/Dockerfile already do). Postgres
must be reachable from the host (Compose service name ``db`` does not resolve
here). Publish 5432 on loopback (``docker compose up -d db`` after the compose
port mapping) or set ``AI_AGENT_STUDIO_DB_HOST=127.0.0.1``.

    pip install "langgraph-cli[inmem]"
    langgraph dev

Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
CopilotKit AG-UI: http://127.0.0.1:2024/copilotkit
Production: https://agent.<DOMAIN_NAME>/copilotkit

Every request except GET /ok requires the product JWT:

    Authorization: Bearer <access_token>

``disable_studio_auth`` in langgraph.json (and the Agent Server image) turns off
LangSmith Studio login so this JWT is the only gate. It is not "no auth".
Desktop Studio (``LANGSMITH_LANGGRAPH_DESKTOP``) and ``AI_AGENT_STUDIO_USER_PHONE``
impersonation are refused when Django ``DEBUG`` is false.

Do not pass configurable.user_id from the browser. Tools run as the JWT user.
List and resume conversations with @langchain/langgraph-sdk against the same
host (threads.search, threads.getHistory) using the same Bearer token.
"""

from __future__ import annotations

import os

import django

from ai_agent.conf import resolve_studio_django_settings_module

# Prefer AI_AGENT_STUDIO_DJANGO_SETTINGS, else keep DJANGO_SETTINGS_MODULE.
# Host images/compose set DJANGO_SETTINGS_MODULE; do not default to a product.
os.environ["DJANGO_SETTINGS_MODULE"] = resolve_studio_django_settings_module()
django.setup()

from ai_agent.conf import assert_production_studio_safe, get_agent_settings  # noqa: E402

assert_production_studio_safe()

from ai_agent.graph import build_copilotkit_http_app  # noqa: E402
from ai_agent.runtime import AgentRuntime  # noqa: E402

agent_runtime = AgentRuntime.create(
    settings=get_agent_settings(),
    checkpointer=None,
    store=None,
)
graph = agent_runtime.supervisor()
memory = agent_runtime.memory_graph()
memory_curator = agent_runtime.curator_graph()
app = build_copilotkit_http_app(graph)
