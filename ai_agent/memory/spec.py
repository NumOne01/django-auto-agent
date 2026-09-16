"""Resolve AppConfig memory schemas for the supervisor and each Django app."""

from __future__ import annotations

from dataclasses import dataclass

from django.apps import apps
from django.utils.module_loading import import_string
from pydantic import BaseModel

from ai_agent.memory.namespaces import SUPERVISOR_LAYER
from ai_agent.memory.schemas import Episode, Profile, SemanticFact


@dataclass(frozen=True)
class MemorySpec:
    layer: str
    profile: type[BaseModel]
    collections: tuple[type[BaseModel], ...]
    episode: type[BaseModel]


def resolve_memory_spec(layer: str) -> MemorySpec:
    profile = Profile
    collections: tuple[type[BaseModel], ...] = (SemanticFact,)
    episode = Episode
    app_label = "ai_agent" if layer in {SUPERVISOR_LAYER, "{memory_layer}"} else layer
    try:
        config = apps.get_app_config(app_label)
    except LookupError:
        config = None
    if config is not None:
        profile = _resolve_model(
            getattr(config, "agent_memory_profile", None), profile
        )
        raw_collections = getattr(config, "agent_memory_collections", None)
        if raw_collections:
            resolved = tuple(
                _resolve_model(item, SemanticFact) for item in raw_collections
            )
            if resolved:
                collections = resolved
        episode = _resolve_model(
            getattr(config, "agent_memory_episode", None), episode
        )
    extra = _model_agent_for_layer(layer)
    if extra is not None:
        profile = _resolve_model(getattr(extra, "memory_profile", None), profile)
        raw_collections = getattr(extra, "memory_collections", None)
        if raw_collections:
            resolved = tuple(
                _resolve_model(item, SemanticFact) for item in raw_collections
            )
            if resolved:
                collections = resolved
        episode = _resolve_model(getattr(extra, "memory_episode", None), episode)
    return MemorySpec(
        layer=layer,
        profile=profile,
        collections=collections,
        episode=episode,
    )


def _model_agent_for_layer(layer: str):
    from ai_agent.agents import model_agent_for_layer

    return model_agent_for_layer(layer)


def _resolve_model(value, default: type[BaseModel]) -> type[BaseModel]:
    if value is None:
        return default
    if isinstance(value, str):
        value = import_string(value)
    if isinstance(value, type) and issubclass(value, BaseModel):
        return value
    return default
