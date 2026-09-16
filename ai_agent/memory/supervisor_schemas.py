"""Cross-domain schemas for the supervisor's overall memory."""

from typing import ClassVar, Optional

from pydantic import BaseModel, Field

from ai_agent.memory.schemas import Episode, SemanticFact


class SupervisorProfile(BaseModel):
    """Standing identity for the overall assistant. Not the current task."""

    name: Optional[str] = Field(
        default=None,
        description="The customer's preferred name, if known.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Preferred language for replies across domains.",
    )
    reply_style: Optional[str] = Field(
        default=None,
        description=(
            "Standing how-to-address rule, e.g. start replies with their name. "
            "One short rule. Not the current request."
        ),
    )
    preferred_domains: Optional[str] = Field(
        default=None,
        description=(
            "App labels this customer typically uses. Not the current request."
        ),
    )
    standing_constraints: Optional[str] = Field(
        default=None,
        description=(
            "Global rules that should influence most future conversations, "
            "e.g. never suggest a product they declined, always confirm "
            "mutating actions. Not a log of the current task."
        ),
    )


class SupervisorFact(SemanticFact):
    """Cross-domain durable facts that do not belong on the supervisor profile."""


class DomainHabit(BaseModel):
    """How the customer typically moves across specialists."""

    natural_key: ClassVar[tuple[str, ...]] = ("domain",)

    domain: Optional[str] = Field(
        default=None,
        description="App label this habit is about.",
    )
    typical_ask: Optional[str] = Field(
        default=None,
        description=(
            "What they usually want in that domain, e.g. checks one specialist "
            "then another."
        ),
    )


class SupervisorEpisode(Episode):
    """Routing and conversation traces that help the supervisor next time."""
