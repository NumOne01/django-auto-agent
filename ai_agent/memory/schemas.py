"""Default Pydantic schemas for LangMem profile, semantic, and episodic memory."""

from typing import ClassVar, Optional

from pydantic import BaseModel, Field


class Profile(BaseModel):
    """Fallback singleton for a layer that does not define its own profile schema."""

    goals: Optional[str] = Field(
        default=None,
        description="Current main goals for this capability.",
    )
    preferences: Optional[str] = Field(
        default=None,
        description=(
            "How this capability should behave: tone, language, or constraints "
            "that apply here."
        ),
    )
    current_state: Optional[str] = Field(
        default=None,
        description=(
            "Standing working picture for this capability only. "
            "Not a transcript of the current turn, date range, or one-off request."
        ),
    )


class SemanticFact(BaseModel):
    """A durable fact or preference with isolating context."""

    natural_key: ClassVar[tuple[str, ...]] = ("subject", "predicate")

    subject: str = Field(description="Who or what this fact is about.")
    predicate: str = Field(description="The relationship or attribute.")
    object: str = Field(description="The value or related entity.")
    context: Optional[str] = Field(
        default=None,
        description="When this fact applies, so it stays useful out of context.",
    )


class Episode(BaseModel):
    """A reusable problem-solving trace from the agent's perspective."""

    observation: str = Field(
        description="The context and setup — what the user needed."
    )
    thoughts: str = Field(description="Internal reasoning. Written as 'I ...'.")
    action: str = Field(description="What was done, how, and in what format.")
    result: str = Field(
        description=(
            "Outcome and what made it worth remembering. If it failed, include "
            "the reason and how a similar failure could be prevented next time."
        )
    )


class PlaybookRule(BaseModel):
    """One standing how-to for this user and layer. Not a transcript."""

    trigger: str = Field(description="Situation when this rule applies.")
    do: str = Field(description="What to do in that situation.")
    dont: Optional[str] = Field(
        default=None, description="What to avoid, if relevant."
    )
    why: Optional[str] = Field(
        default=None, description="Why this rule exists for this customer."
    )


class LayerPlaybook(BaseModel):
    """Curator-owned singleton: a short local policy for this user and layer."""

    rules: list[PlaybookRule] = Field(
        default_factory=list,
        description="At most eight standing rules. Oldest extra rules are dropped.",
    )
