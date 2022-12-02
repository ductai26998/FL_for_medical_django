# import requests
# from rest_framework import status, views
# from rest_framework.response import Response
# from ..center_training.utils import train_center

# from .. import CLIENT_API_URL
# from ..center.models import CenterEvent
# from ..client.models import Client
# from . import ErrorCode, EventType
# from .serializers import (CenterReceivesParamsInputSerializer,
#                           CenterSendsParamsInputSerializer)


# class CenterSendsParams(views.APIView):
#     @classmethod
#     def post(self, request, **kwargs):
#         data = request.data
#         serializer = CenterSendsParamsInputSerializer(data=data)
#         if serializer.is_valid():
#             serializer.validated_data

#             client_ids = Client.objects.values_list("id", flat=True)
#             for client_id in client_ids:
#                 global_round = data["global_round"]
#                 model_path = data["model_path"]
#                 ## STEP 3: Center send params to clients
#                 print("STEP 3")
#                 CenterEvent.objects.create(event_type=EventType.CENTER_SENT_PARAMS,
#                                                client_id=client_id, global_round=global_round, model_path=model_path)
#                 res = requests.post(
#                     CLIENT_API_URL + "/client/params/receives", json={"global_round": global_round, "model_path": model_path}
#                 )
#                 print("/client/params/receives", res.json())
#                 # if res.ok:
#                 #     # FIXME: divide to fail case and success case with is_success field
#             return Response(
#                 {
#                     "detail": "Sent params to clients successfully"
#                 },
#                 status=status.HTTP_200_OK,
#             )

#         return Response(
#             {
#                 "code": ErrorCode.PROCESSING_ERROR,
#                 "detail": "Can not send params to clients",
#                 "messages": serializer.errors,
#             },
#             status=status.HTTP_400_BAD_REQUEST,
#         )


# class CenterReceivesParams(views.APIView):
#     @classmethod
#     def post(self, request, **kwargs):
#         ## STEP 8: Center receives params from client
#         print("STEP 8")
#         data = request.data
#         client_id = data["client_id"]
#         serializer = CenterReceivesParamsInputSerializer(data=data)
#         if serializer.is_valid():
#             serializer.validated_data
#             client_ids = Client.objects.values_list("id", flat=True)
#             global_round = data["global_round"]
#             event_clients = CenterEvent.objects.filter(
#                 event_type=EventType.CENTER_RECEIVED_PARAMS, global_round=global_round).values_list("client_id", flat=True)
#             if client_id not in client_ids or len(event_clients) >= len(client_ids):
#                 return Response(
#                     {
#                         "code": ErrorCode.PERMISSION_DENIED,
#                         "detail": "permission denied",
#                     },
#                     status=status.HTTP_400_BAD_REQUEST,
#                 )
#             if client_id in event_clients:
#                 return Response(
#                     {
#                         "code": ErrorCode.EXISTED,
#                         "detail": "This client sent params before",
#                     },
#                     status=status.HTTP_400_BAD_REQUEST,
#                 )
#             if len(event_clients) >= len(client_ids) - 1:
#                 # STEP 9: Center calculate params
#                 print("STEP 9")
#                 train_center(global_round + 1)
#             CenterEvent.objects.create(
#                 event_type=EventType.CENTER_RECEIVED_PARAMS, **data)
#             return Response(
#                 {
#                     "detail": "Received params from client %s successfully" % client_id,
#                 },
#                 status=status.HTTP_200_OK,
#             )

#         return Response(
#             {
#                 "code": ErrorCode.PROCESSING_ERROR,
#                 "detail": "Can not receive params from client: %s" % client_id,
#                 "messages": serializer.errors,
#             },
#             status=status.HTTP_400_BAD_REQUEST,
#         )


# class CenterGetClientParams(views.APIView):
#     @classmethod
#     def get(self, request, **kwargs):
#         data = request.data
#         global_round = data.get("global_round")
#         if not global_round:
#             return Response(
#                 {
#                     "code": ErrorCode.REQUIRED,
#                     "detail": "global_round is required",
#                 },
#                 status=status.HTTP_400_BAD_REQUEST,
#             )
#         model_path_list = CenterEvent.objects.filter(
#             event_type=EventType.CENTER_SENT_PARAMS,
#             global_round=global_round
#         ).values_list("model_path", flat=True)
#         return Response(
#             {
#                 "data": model_path_list
#             },
#             status=status.HTTP_200_OK,
#         )



# import requests
# from rest_framework import status, views
# from rest_framework.response import Response
# from ..client_training.utils import train_client

# from .. import CENTER_API_URL
# from . import ErrorCode
# from .serializers import (ClientReceivesParamsInputSerializer,
#                           ClientSendsParamsInputSerializer)

# CLIENT_ID = 1


# class ClientSendsParams(views.APIView):
#     @classmethod
#     def post(self, request, **kwargs):
#         data = request.data
#         serializer = ClientSendsParamsInputSerializer(data=data)
#         if serializer.is_valid():
#             serializer.validated_data
#             ## STEP 7: Client sends params to center
#             print("STEP 7")
#             global_round = data["global_round"]
#             model_path = data["model_path"]
#             res = requests.post(
#                 CENTER_API_URL + "/center/params/receives", json={"client_id": CLIENT_ID, "global_round": global_round, "model_path": model_path}
#             )
#             print("/center/params/receives", res.json())
#             # FIXME: if send params fail -> resend
#             return Response(
#                 {
#                     "detail": "Sent params to center successfully"
#                 },
#                 status=status.HTTP_200_OK,
#             )

#         return Response(
#             {
#                 "code": ErrorCode.PROCESSING_ERROR,
#                 "detail": "Can not send params to center",
#                 "messages": serializer.errors,
#             },
#             status=status.HTTP_400_BAD_REQUEST,
#         )


# class ClientReceivesParams(views.APIView):
#     @classmethod
#     def post(self, request, **kwargs):
#         ## STEP 4: Client received params from center
#         print("STEP 4")
#         data = request.data
#         serializer = ClientReceivesParamsInputSerializer(data=data)
#         if serializer.is_valid():
#             serializer.validated_data
#             global_round = data["global_round"]
#             model_path = data["model_path"]
#             ## STEP 5: Client trains its model with above params
#             print("STEP 5")
#             train_client(global_round, model_path)
#             return Response(
#                 {
#                     "detail": "Received params from center successfully",
#                 },
#                 status=status.HTTP_200_OK,
#             )

#         return Response(
#             {
#                 "code": ErrorCode.PROCESSING_ERROR,
#                 "detail": "Can not receive params from center",
#                 "messages": serializer.errors,
#             },
#             status=status.HTTP_400_BAD_REQUEST,
#         )
