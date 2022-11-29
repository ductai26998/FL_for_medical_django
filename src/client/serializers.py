from rest_framework import serializers


class ClientSendsParamsInputSerializer(serializers.Serializer):
    global_round = serializers.IntegerField(required=True)
    model_path = serializers.CharField(required=True)


class ClientReceivesParamsInputSerializer(serializers.Serializer):
    global_round = serializers.IntegerField(required=True)
    model_path = serializers.CharField(required=True)
