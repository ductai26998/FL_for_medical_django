import requests
from django.conf import settings
from django.utils import timezone
from rest_framework import status, views
from rest_framework.response import Response

from .. import CENTER_API_URL
from ..client_training.utils import train_client
from . import ErrorCode
from .serializers import (ClientReceivesParamsInputSerializer,
                          ClientSendsParamsInputSerializer)


class ClientSendsParams(views.APIView):
    @classmethod
    def post(self, request, **kwargs):
        data = request.data
        serializer = ClientSendsParamsInputSerializer(data=data)
        if serializer.is_valid():
            serializer.validated_data
            # STEP 7: Client sends params to center
            print("STEP 7", timezone.now())
            global_round = data["global_round"]
            model_path = data["model_path"]
            res = requests.post(
                CENTER_API_URL + "/center/params/receives", json={"client_id": settings.CLIENT_ID, "global_round": global_round, "model_path": model_path}
            )
            print("/center/params/receives", res.json())
            # FIXME: if send params fail -> resend
            return Response(
                {
                    "detail": "Sent params to center successfully"
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "code": ErrorCode.PROCESSING_ERROR,
                "detail": "Can not send params to center",
                "messages": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class ClientReceivesParams(views.APIView):
    @classmethod
    def post(self, request, **kwargs):
        # STEP 4: Client received params from center
        print("STEP 4", timezone.now())
        data = request.data
        serializer = ClientReceivesParamsInputSerializer(data=data)
        if serializer.is_valid():
            serializer.validated_data
            global_round = data["global_round"]
            model_path = data["model_path"]
            # STEP 5: Client trains its model with above params
            print("STEP 5", timezone.now())
            train_client(global_round, model_path)
            return Response(
                {
                    "detail": "Received params from center successfully",
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "code": ErrorCode.PROCESSING_ERROR,
                "detail": "Can not receive params from center",
                "messages": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )
