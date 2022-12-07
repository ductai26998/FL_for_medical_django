from rest_framework import serializers

from .models import Event


class ClientSendsParamsInputSerializer(serializers.Serializer):
    global_round = serializers.IntegerField(required=True)
    model_path = serializers.CharField(required=True)


class ClientReceivesParamsInputSerializer(serializers.Serializer):
    global_round = serializers.IntegerField(required=True)
    model_path = serializers.CharField(required=True)


class CenterSendsParamsInputSerializer(serializers.Serializer):
    global_round = serializers.IntegerField(required=True)
    model_path = serializers.CharField(required=True)


class CenterEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = Event
        fields = "__all__"


class CenterReceivesParamsInputSerializer(serializers.Serializer):
    client_id = serializers.IntegerField(required=True)
    global_round = serializers.IntegerField(required=True)
    model_path = serializers.CharField(required=True)


class PredictInputSerializer(serializers.Serializer):
    images = serializers.ListField(child=serializers.FileField(allow_empty_file=False))
