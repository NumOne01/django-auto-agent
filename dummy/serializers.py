from rest_framework import serializers

from dummy.models import Item


class ItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Item
        fields = ("id", "name", "secret", "status", "quantity")
        read_only_fields = ("id",)


class ItemCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=64)
    secret = serializers.CharField(max_length=64, required=False, allow_blank=True)
    status = serializers.ChoiceField(choices=Item.STATUS_CHOICES, required=False)
    quantity = serializers.DecimalField(
        max_digits=12, decimal_places=4, required=False
    )


class ItemPatchSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=64, required=False)
    status = serializers.ChoiceField(choices=Item.STATUS_CHOICES, required=False)
    allocations = serializers.JSONField(required=False)


class TicketCreateSerializer(serializers.Serializer):
    category = serializers.IntegerField()
    message = serializers.CharField()
    attachments = serializers.ListField(
        child=serializers.FileField(), required=False
    )
