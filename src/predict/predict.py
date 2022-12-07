from ..utils.aws_s3 import read_params_from_s3
from ..train.models import CNNModel
from ..device.models import Device
from django.conf import settings


def predict(image_path):
    client = Device.objects.get(id=settings.CLIENT_ID)
    model_path = client.current_model_path
    params = read_params_from_s3(model_path)
    model = CNNModel(num_channels=client.num_channels, num_classes=client.num_classes)
    model.set_weights(params)
    res = model.predict([image_path])
    print(res)
    return res
