from django.db import connection
from django.db.models import F, Max, OuterRef, Subquery
from django.http import Http404
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ai_agent.expose import agent_exclude
from ai_agent.memory.user_memories import (
    delete_for_user,
    get_for_user,
    list_for_user,
)
from ai_agent.models import AgentMessage
from ai_agent.serializers import (
    AgentMemorySerializer,
    AgentMessageSerializer,
    AgentThreadSummarySerializer,
)


@extend_schema(
    operation_id="list_agent_threads",
    summary="List agent chat threads",
    description=(
        "List this user's product chat threads. History comes from AgentMessage, "
        "not compacted LangGraph checkpoints."
    ),
    tags=["AI Agent"],
    responses={200: AgentThreadSummarySerializer(many=True)},
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@agent_exclude
def list_agent_threads(request):
    threads = _thread_summaries(request.user)
    serializer = AgentThreadSummarySerializer(threads, many=True)
    return Response({"threads": serializer.data})


def _thread_summaries(user):
    """Latest message per thread. DISTINCT ON on Postgres; subquery elsewhere."""
    qs = AgentMessage.objects.filter(user=user)
    if connection.features.can_distinct_on_fields:
        latest_pks = (
            qs.order_by("thread_id", "-created_at", "-id")
            .distinct("thread_id")
            .values("pk")
        )
        return (
            AgentMessage.objects.filter(pk__in=Subquery(latest_pks))
            .annotate(updated_at=F("created_at"), preview=F("content"))
            .values("thread_id", "updated_at", "preview")
            .order_by("-updated_at")
        )
    latest = qs.filter(thread_id=OuterRef("thread_id")).order_by(
        "-created_at", "-id"
    )
    return (
        qs.values("thread_id")
        .annotate(
            updated_at=Max("created_at"),
            preview=Subquery(latest.values("content")[:1]),
        )
        .order_by("-updated_at")
    )


@extend_schema(
    operation_id="list_agent_thread_messages",
    summary="List agent chat messages",
    description=(
        "List user-visible messages for one of this user's threads. "
        "Does not include compaction summaries or tool calls."
    ),
    tags=["AI Agent"],
    responses={200: AgentMessageSerializer(many=True)},
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@agent_exclude
def list_agent_thread_messages(request, thread_id):
    if not AgentMessage.objects.filter(
        user=request.user, thread_id=thread_id
    ).exists():
        raise Http404("Thread not found")
    messages = AgentMessage.objects.filter(
        user=request.user, thread_id=thread_id
    ).order_by("created_at", "id")
    serializer = AgentMessageSerializer(messages, many=True)
    return Response({"messages": serializer.data})


@extend_schema(
    operation_id="list_agent_memories",
    summary="List agent memories",
    description=(
        "List this user's profile and semantic memories from the LangGraph store. "
        "Episodes, playbooks, and prompt overlays are not included."
    ),
    tags=["AI Agent"],
    parameters=[
        OpenApiParameter(
            name="layer",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Optional memory layer, e.g. supervisor or a host app label.",
        ),
    ],
    responses={200: AgentMemorySerializer(many=True)},
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@agent_exclude
def list_agent_memories(request):
    layer = (request.query_params.get("layer") or "").strip()
    memories, error = list_for_user(str(request.user.pk), layer=layer)
    unavailable = _memory_unavailable(error)
    if unavailable is not None:
        return unavailable
    serializer = AgentMemorySerializer(memories, many=True)
    return Response({"memories": serializer.data})


@extend_schema(
    methods=["GET"],
    operation_id="get_agent_memory",
    summary="Get an agent memory",
    description="Retrieve one of this user's profile or semantic memories.",
    tags=["AI Agent"],
    responses={200: AgentMemorySerializer},
)
@extend_schema(
    methods=["DELETE"],
    operation_id="delete_agent_memory",
    summary="Delete an agent memory",
    description="Delete one of this user's profile or semantic memories.",
    tags=["AI Agent"],
    request=None,
    responses={204: None},
)
@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
@agent_exclude
def agent_memory_detail(request, layer, kind, key):
    user_id = str(request.user.pk)
    if request.method == "DELETE":
        deleted, error = delete_for_user(
            user_id, layer=layer, kind=kind, key=key
        )
        unavailable = _memory_unavailable(error)
        if unavailable is not None:
            return unavailable
        if not deleted:
            raise Http404("Memory not found")
        return Response(status=status.HTTP_204_NO_CONTENT)

    memory, error = get_for_user(user_id, layer=layer, kind=kind, key=key)
    unavailable = _memory_unavailable(error)
    if unavailable is not None:
        return unavailable
    if memory is None:
        raise Http404("Memory not found")
    serializer = AgentMemorySerializer(memory)
    return Response({"memory": serializer.data})


def _memory_unavailable(error):
    if not error:
        return None
    return Response(
        {"detail": "Agent memory store is unavailable"},
        status=status.HTTP_503_SERVICE_UNAVAILABLE,
    )
