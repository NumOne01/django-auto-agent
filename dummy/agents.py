from dummy.models import Item

from ai_agent.agents import ModelAgent, agent_tool
from ai_agent.context import get_current_user


class CatalogSearchAgent(ModelAgent):
    name = "catalog_search"
    description = (
        "Search the authenticated user's catalog items by name without calling the HTTP API."
    )

    @agent_tool
    def find_items(self, query: str) -> str:
        """Find catalog items owned by the current user whose name contains query."""
        user = get_current_user()
        names = list(
            Item.objects.filter(owner=user, name__icontains=query).values_list(
                "name", flat=True
            )
        )
        if not names:
            return "No matching items."
        return ", ".join(names)
