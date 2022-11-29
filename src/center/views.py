import requests
from rest_framework import status, views
from rest_framework.response import Response
from ..center_training.utils import train_center

from .. import CLIENT_API_URL
from ..center.models import CenterEvent
from ..client.models import Client
from . import ErrorCode, EventType
from .serializers import (CenterReceivesParamsInputSerializer,
                          CenterSendsParamsInputSerializer)


class CenterSendsParams(views.APIView):
    @classmethod
    def post(self, request, **kwargs):
        data = request.data
        serializer = CenterSendsParamsInputSerializer(data=data)
        if serializer.is_valid():
            serializer.validated_data

            client_ids = Client.objects.values_list("id", flat=True)
            for client_id in client_ids:
                global_round = data["global_round"]
                model_path = data["model_path"]
                ## STEP 3: Center send params to clients
                print("STEP 3")
                res = requests.post(
                    # FIXME: update URL -> client URL
                    CLIENT_API_URL + "/client/params/receives", json={"global_round": global_round, "model_path": model_path}
                )
                if res.ok:
                    ## STEP 4: Check send params to clients success or not
                    print("STEP 4")
                    # create event
                    # FIXME: divide to fail case and success case with is_success field
                    CenterEvent.objects.create(event_type=EventType.CENTER_SENT_PARAMS,
                                               client_id=client_id, global_round=global_round, model_path=model_path)
            return Response(
                {
                    "detail": "Sent params to clients successfully"
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "code": ErrorCode.PROCESSING_ERROR,
                "detail": "Can not send params to clients",
                "messages": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class CenterReceivesParams(views.APIView):
    @classmethod
    def post(self, request, **kwargs):
        ## STEP 12: Center receives params from client
        print("STEP 12")
        data = request.data
        client_id = data["client_id"]
        serializer = CenterReceivesParamsInputSerializer(data=data)
        if serializer.is_valid():
            serializer.validated_data
            client_ids = Client.objects.values_list("id", flat=True)
            global_round = data["global_round"]
            event_clients = CenterEvent.objects.filter(
                event_type=EventType.CENTER_RECEIVED_PARAMS, global_round=global_round).values_list("client_id", flat=True)
            if client_id not in client_ids or len(event_clients) >= len(client_ids):
                return Response(
                    {
                        "code": ErrorCode.PERMISSION_DENIED,
                        "detail": "permission denied",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if client_id in event_clients:
                return Response(
                    {
                        "code": ErrorCode.EXISTED,
                        "detail": "This client sent params before",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if len(event_clients) >= len(client_ids) - 1:
                # STEP 13: Center calculate params
                print("STEP 13")
                train_center(global_round + 1)
            CenterEvent.objects.create(
                event_type=EventType.CENTER_RECEIVED_PARAMS, **data)
            return Response(
                {
                    "detail": "Received params from client %s successfully" % client_id,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "code": ErrorCode.PROCESSING_ERROR,
                "detail": "Can not receive params from client: %s" % client_id,
                "messages": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class CenterGetClientParams(views.APIView):
    @classmethod
    def get(self, request, **kwargs):
        data = request.data
        global_round = data.get("global_round")
        if not global_round:
            return Response(
                {
                    "code": ErrorCode.REQUIRED,
                    "detail": "global_round is required",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        model_path_list = CenterEvent.objects.filter(
            event_type=EventType.CENTER_SENT_PARAMS,
            global_round=global_round
        ).values_list("model_path", flat=True)
        return Response(
            {
                "data": model_path_list
            },
            status=status.HTTP_200_OK,
        )
