from typing import ClassVar, Optional

from pydantic import BaseModel, Field


class DummyProfile(BaseModel):
    """Standing preferences for the dummy catalog specialist."""

    language: Optional[str] = Field(
        default=None,
        description="Preferred language for catalog replies.",
    )
    standing_pref: Optional[str] = Field(
        default=None,
        description="How this specialist should behave for this customer.",
    )


class DummyAccount(BaseModel):
    """A named account the customer uses with this catalog."""

    natural_key: ClassVar[tuple[str, ...]] = ("bank_name", "label")

    bank_name: Optional[str] = Field(default=None, description="Institution name.")
    label: Optional[str] = Field(default=None, description="Nickname for this account.")
    use_for: Optional[str] = Field(
        default=None,
        description="What this account is typically used for.",
    )
