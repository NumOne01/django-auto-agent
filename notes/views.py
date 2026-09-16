from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ai_agent.expose import agent_expose


@agent_expose
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def public_note(request):
    return Response({"note": "visible to the agent"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def private_note(request):
    return Response({"note": "hidden from the agent"})
