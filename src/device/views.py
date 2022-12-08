from concurrent.futures import ProcessPoolExecutor

import requests
from django.conf import settings
from django.utils import timezone
from rest_framework import status, views
from rest_framework.response import Response

from ..train.models import CNNModel
from ..train.utils import train_center, train_client
from ..utils.aws_s3 import read_params_from_s3
from . import ErrorCode, EventStatus, EventType
from .models import Device, Event
from .serializers import (
    CenterReceivesParamsInputSerializer,
    CenterSendsParamsInputSerializer,
    ClientReceivesParamsInputSerializer,
    ClientSendsParamsInputSerializer,
    PredictInputSerializer,
)


def send_params_to_client(client, global_round, model_path):
    print("Send params to client %s with url %s" % (client.id, client.api_url))
    Event.objects.create(
        event_type=EventType.CENTER_SENT_PARAMS,
        device_id=client.id,
        global_round=global_round,
        model_path=model_path,
        status=EventStatus.STARTED,
    )
    res = requests.post(
        client.api_url + "/client/params/receives",
        json={"global_round": global_round, "model_path": model_path},
    )
    print("/client/params/receives", res.json())


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
            center.save(update_fields=["current_global_round", "current_model_path"])
            clients = (
                Device.objects.filter(is_center=False,).exclude(
                    events__global_round=global_round,
                    events__event_type=EventType.CENTER_SENT_PARAMS,
                    events__status=EventStatus.COMPLETED,
                )
                # .distinct("api_url")
            )
            # STEP 3: Center send params to clients
            print("STEP 3", timezone.now())

            with ProcessPoolExecutor() as executor:
                for client in clients:
                    executor.submit(
                        send_params_to_client, client, global_round, model_path
                    )
                print("Sent params to clients")
                return Response(
                    {"detail": "Sent params to clients successfully"},
                    status=status.HTTP_200_OK,
                )
            # FIXME: divide to fail case and success case with is_success field

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
            client_ids = Device.objects.filter(is_center=False).values_list(
                "id", flat=True
            )
            global_round = data["global_round"]
            event_clients = Event.objects.filter(
                event_type=EventType.CENTER_RECEIVED_PARAMS,
                global_round=global_round,
                status=EventStatus.COMPLETED,
            ).values_list("device_id", flat=True)
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
                model_path=model_path,
                status=EventStatus.COMPLETED,
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
        model_path_list = Device.objects.filter(
            is_center=False, current_global_round=global_round
        ).values_list("model_path", flat=True)
        return Response(
            {"data": model_path_list},
            status=status.HTTP_200_OK,
        )


def send_params_to_center(center, global_round, model_path):
    res = requests.post(
        center.api_url + "/center/params/receives",
        json={
            "client_id": settings.CLIENT_ID,
            "global_round": global_round,
            "model_path": model_path,
        },
    )
    print("/center/params/receives", res.json())


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
            with ProcessPoolExecutor() as executor:
                executor.submit(send_params_to_center, center, global_round, model_path)
                return Response(
                    {"detail": "Sent params to center successfully"},
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
            center_event = Event.objects.filter(
                global_round=global_round,
                event_type=EventType.CENTER_SENT_PARAMS,
                device_id=settings.CLIENT_ID,
            ).first()
            center_event.status = EventStatus.COMPLETED
            center_event.save(update_fields=["status"])
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

from django.shortcuts import render
class CenterPredict(views.APIView):
    @classmethod
    def get(self, request, **kwargs):
        return render(request, "pages/predict/index.html")

    @classmethod
    def post(self, request, **kwargs):
        data = request.data
        serializer = PredictInputSerializer(data=data)
        if serializer.is_valid():
            serializer.validated_data
            images = request.FILES.getlist("images")
            center = Device.objects.get(is_center=True)
            model_path = center.current_model_path
            params = read_params_from_s3(model_path)
            model = CNNModel(
                num_channels=center.num_channels, num_classes=center.num_classes
            )
            model.set_weights(params)
            res = model.predict_files(images)
            return Response(
                {"detail": "Success", "data": res},
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "code": ErrorCode.PROCESSING_ERROR,
                "detail": "Can not predict",
                "messages": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class ClientPredict(views.APIView):
    @classmethod
    def post(self, request, **kwargs):
        data = request.data
        serializer = PredictInputSerializer(data=data)
        if serializer.is_valid():
            serializer.validated_data
            images = request.FILES.getlist("images")
            client = Device.objects.get(id=settings.CLIENT_ID)
            model_path = client.current_model_path
            params = read_params_from_s3(model_path)
            model = CNNModel(
                num_channels=client.num_channels, num_classes=client.num_classes
            )
            model.set_weights(params)
            res = model.predict_files(images)
            return Response(
                {"detail": "Success", "data": res},
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "code": ErrorCode.PROCESSING_ERROR,
                "detail": "Can not predict",
                "messages": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )
