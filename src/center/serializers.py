from rest_framework import serializers
from .models import CenterEvent


class CenterSendsParamsInputSerializer(serializers.Serializer):
    global_round = serializers.IntegerField(required=True)
    params = serializers.JSONField(required=True)


class CenterEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = CenterEvent
        fields = "__all__"


class CenterReceivesParamsInputSerializer(serializers.Serializer):
    client_id = serializers.IntegerField(required=True)
    global_round = serializers.IntegerField(required=True)
    params = serializers.JSONField(required=True)
