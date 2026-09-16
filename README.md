# django-auto-agent

Django reusable app that turns opted-in DRF endpoints into a LangGraph supervisor, with optional long-term memory, admin views, and evals.

Import name: `ai_agent`.

## Install

```bash
pip install -e .
```

Redis-backed memory store (optional):

```bash
pip install -e ".[redis]"
```

## Host wiring

```python
INSTALLED_APPS = [
    # ...
    "ai_agent",
    "catalog",  # your domain app
]

AI_AGENT = {
    "SUPERVISOR_MODEL": "openai:gpt-4.1-mini",
    "SUBAGENT_MODEL": "openai:gpt-4.1-nano",
    "AUTHENTICATE_TOKEN": "myproject.auth.authenticate_access_token",
    "PLATFORM_NAME": "Acme",
    "PLATFORM_DOMAINS": ("catalog", "billing"),
    "EVAL_MODULE": "catalog.evals",
}
```

`AUTHENTICATE_TOKEN` is required. It must be a callable (or dotted path to one) that accepts a bearer token and returns a user object with `.pk`.

Optional evals — point `EVAL_MODULE` at a host module that defines:

- `seed_eval_world() -> EvalWorld` (users plus `prompt_vars` for case prompts)
- `snapshot_world(world, kind) -> dict` (optional; used by mutation cases)
- `CASES: list[EvalCase]` (or partitioned `ROUTING_CASES` / `TOOL_CASES` / `MUTATION_CASES` / `E2E_CASES`)

The library always includes safety routing cases (off-policy greeting, jailbreaks). Domain tool and mutation cases come from the host module. See `dummy/evals.py` in this repo.

`HTTP_TRUSTED_PROXY_COUNT` (default `0`) controls whether the CopilotKit HTTP throttle reads `X-Forwarded-For`. Leave it at `0` unless the app sits behind that many trusted reverse proxies; otherwise clients can rotate a spoofed header and bypass the limit.

Include the chat/memory HTTP API:

```python
urlpatterns = [
    path("api/agent/", include("ai_agent.urls")),
]
```

Opt a host app into the agent by setting flags on its `AppConfig`:

```python
class CatalogConfig(AppConfig):
    name = "catalog"
    agent_expose = True
    agent_description = "List and create catalog items for the authenticated user."
    agent_exclude = ("internal_webhook",)
    agent_memory_profile = "catalog.memory_schemas.CatalogProfile"
    agent_memory_collections = (
        "ai_agent.memory.schemas.SemanticFact",
        "catalog.memory_schemas.CatalogAccount",
    )
```

Mark individual views even when the app is off:

```python
from ai_agent.expose import agent_expose, agent_exclude

@agent_expose
def public_note(request):
    ...
```

## LangGraph Agent Server

This package is not a host Docker image. Install `django-auto-agent` **and** your Django project into the Agent Server image, then point `langgraph.json` at the library entrypoints (see the example `langgraph.json` in this repo):

- graphs: `ai_agent.studio:graph`, `ai_agent.studio:memory`
- auth: `ai_agent.auth:auth`
- HTTP app: `ai_agent.studio:app`

Set `DJANGO_SETTINGS_MODULE` (or `AI_AGENT_STUDIO_DJANGO_SETTINGS`) to **your** settings module before starting Studio.

The example `langgraph.json` uses `cors.allow_origins: ["*"]`. Auth is Bearer-token based (not cookies), so CSRF exposure is limited; restrict origins in production.

## Tests

From this repo:

```bash
python manage.py test ai_agent
```

The `dummy` and `notes` apps plus `tests/settings.py` are a sample host. They are not installed with the wheel.
