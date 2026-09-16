from drf_spectacular.utils import extend_schema
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ai_agent.expose import agent_exclude
from dummy.models import Item
from dummy.serializers import (
    ItemCreateSerializer,
    ItemPatchSerializer,
    ItemSerializer,
    TicketCreateSerializer,
)


@extend_schema(responses=ItemSerializer)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_items(request):
    items = Item.objects.filter(owner=request.user)
    return Response(ItemSerializer(items, many=True).data)


@extend_schema(request=ItemCreateSerializer, responses=ItemSerializer)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_item(request):
    serializer = ItemCreateSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    item = Item.objects.create(owner=request.user, **serializer.validated_data)
    return Response(ItemSerializer(item).data, status=201)


@extend_schema(methods=["GET"], responses=ItemSerializer)
@extend_schema(methods=["PATCH"], request=ItemPatchSerializer, responses=ItemSerializer)
@api_view(["GET", "PATCH"])
@permission_classes([IsAuthenticated])
def item_detail(request, pk):
    try:
        item = Item.objects.get(pk=pk, owner=request.user)
    except Item.DoesNotExist:
        return Response({"detail": "Not found."}, status=404)
    if request.method == "PATCH":
        serializer = ItemPatchSerializer(data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        for field, value in serializer.validated_data.items():
            if field == "allocations":
                continue
            setattr(item, field, value)
        item.save()
    return Response(ItemSerializer(item).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dummy_hidden(request):
    return Response({"ok": True})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def dummy_webhook(request):
    return Response({"ok": True})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dummy_internal(request):
    return Response({"ok": True})


@api_view(["POST"])
@parser_classes([MultiPartParser])
@permission_classes([IsAuthenticated])
def dummy_upload(request):
    return Response({"ok": True})


@extend_schema(request=TicketCreateSerializer)
@api_view(["POST"])
@parser_classes([JSONParser, MultiPartParser])
@permission_classes([IsAuthenticated])
def dummy_create_ticket(request):
    serializer = TicketCreateSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    payload = {
        key: value
        for key, value in serializer.validated_data.items()
        if key != "attachments"
    }
    return Response({"id": 1, **payload}, status=201)


dummy_create_ticket.cls.serializer_class = TicketCreateSerializer


@agent_exclude
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dummy_receipt(request):
    return Response({"ok": True})
