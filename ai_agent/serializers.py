from rest_framework import serializers

from ai_agent.models import AgentMessage


class AgentMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentMessage
        fields = (
            "id",
            "external_id",
            "thread_id",
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "created_at",
        )
        read_only_fields = fields


class AgentThreadSummarySerializer(serializers.Serializer):
    thread_id = serializers.CharField()
    updated_at = serializers.DateTimeField()
    preview = serializers.CharField(allow_blank=True)


class AgentMemorySerializer(serializers.Serializer):
    id = serializers.CharField(read_only=True)
    layer = serializers.CharField(read_only=True)
    kind = serializers.CharField(read_only=True)
    key = serializers.CharField(read_only=True)
    content = serializers.JSONField(read_only=True)
    created_at = serializers.DateTimeField(read_only=True, allow_null=True)
    updated_at = serializers.DateTimeField(read_only=True, allow_null=True)
