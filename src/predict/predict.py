import glob

from django.conf import settings

from ..device.models import Device
from ..train.models import CNNModel
from ..utils.aws_s3 import read_params_from_s3
from ..utils.storage import read_params_from_storage


def predict_image_path(image_path):
    client = Device.objects.get(id=settings.CLIENT_ID)
    model_path = client.current_model_path
    if settings.USE_AWS_STORAGE:
        params = read_params_from_s3(model_path)
    else:
        params = read_params_from_storage(model_path)
    model = CNNModel(num_channels=client.num_channels, num_classes=client.num_classes)
    model.set_weights(params)
    res = model.predict([image_path])
    print(res)
    return res


def validate(img_path_regex):
    """
    - img_path_regex: vd: image_path = 'src/train/Dataset/Test/benign/*.jpg'
    """
    breast_img = glob.glob(img_path_regex, recursive=True)
    client = Device.objects.get(id=settings.CLIENT_ID)
    model_path = client.current_model_path
    if settings.USE_AWS_STORAGE:
        params = read_params_from_s3(model_path)
    else:
        params = read_params_from_storage(model_path)
    model = CNNModel(num_channels=client.num_channels, num_classes=client.num_classes)
    model.set_weights(params)
    res = model.predict(breast_img)
    print(res)
    print("total: ", len(res))
    print("total_0: ", len([i for i in res if i == 0]))
    print("total_1: ", len([i for i in res if i == 1]))
