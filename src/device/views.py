import requests
from django.conf import settings
from django.utils import timezone
from rest_framework import status, views
from rest_framework.response import Response

from ..center_training.utils import train_center
from ..client_training.utils import train_client
from . import ErrorCode, EventType, Status
from .models import Device, Event
from .serializers import (CenterReceivesParamsInputSerializer,
                          CenterSendsParamsInputSerializer,
                          ClientReceivesParamsInputSerializer,
                          ClientSendsParamsInputSerializer)


class CenterSendsParams(views.APIView):
    @classmethod
    def post(self, request, **kwargs):
        data = request.data
        serializer = CenterSendsParamsInputSerializer(data=data)
        if serializer.is_valid():
            serializer.validated_data
            global_round = data["global_round"]
            model_path = data["model_path"]
            center = Device.objects.filter(is_center=True).first()
            center.current_model_path = model_path
            center.current_global_round = global_round
            center.save(update_fields=[
                        "current_global_round", "current_model_path"])
            clients = Device.objects.filter(is_center=False)
            for client in clients:
                client.current_global_round = global_round
                client.save(update_fields=["current_global_round"])
                # STEP 3: Center send params to clients
                print("STEP 3", timezone.now())
                Event.objects.create(
                    event_type=EventType.CENTER_SENT_PARAMS,
                    device_id=client.id,
                    global_round=global_round,
                    model_path=model_path,
                    status=Status.STARTED,
                )
                res = requests.post(
                    client.api_url + "/client/params/receives", json={"global_round": global_round, "model_path": model_path}
                )
                print("/client/params/receives", res.json())
                # if res.ok:
                #     # FIXME: divide to fail case and success case with is_success field
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
        # STEP 8: Center receives params from client
        print("STEP 8", timezone.now())
        data = request.data
        client_id = data["client_id"]
        print("client_id", client_id)
        serializer = CenterReceivesParamsInputSerializer(data=data)
        if serializer.is_valid():
            serializer.validated_data
            client_ids = Device.objects.filter(
                is_center=False).values_list("id", flat=True)
            global_round = data["global_round"]
            event_clients = Event.objects.filter(
                event_type=EventType.CENTER_RECEIVED_PARAMS, global_round=global_round).values_list("device_id", flat=True)
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
            model_path = data["model_path"]
            Event.objects.create(
                event_type=EventType.CENTER_RECEIVED_PARAMS,
                device_id=client_id,
                global_round=global_round,
                model_path=model_path
            )
            if len(event_clients) >= len(client_ids) - 1:
                # STEP 9: Center calculate params
                print("STEP 9", timezone.now())
                train_center(global_round + 1)

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
        model_path_list = Event.objects.filter(
            event_type=EventType.CENTER_RECEIVED_PARAMS,
            global_round=global_round
        ).values_list("model_path", flat=True)
        return Response(
            {
                "data": model_path_list
            },
            status=status.HTTP_200_OK,
        )


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
            center = Device.objects.filter(is_center=True).first()
            res = requests.post(
                center.api_url + "/center/params/receives", json={"client_id": settings.CLIENT_ID, "global_round": global_round, "model_path": model_path}
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
