import copy
import glob
import io
import json
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import tensorflow as tf
from django.conf import settings
from django.utils import timezone
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from .. import CENTER_API_URL, CLIENT_API_URL, ROOT_PATH
from ..device.models import Device
from ..utils.aws_s3 import read_params_from_s3, upload_params_to_s3
from .models import CNNModel

# from os import listdir


def get_dataset():
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    img_path = "Dataset/Train/**/*.jpg"
    breast_img = glob.glob(img_path, recursive=True)

    non_can_img = []
    can_img = []

    for img in breast_img:
        img_class = img.split("/")
        if img_class[-2] == "benign":
            non_can_img.append(img)

        elif img_class[-2] == "malignant":
            can_img.append(img)

    non_can_num = len(non_can_img)  # No cancer
    can_num = len(can_img)  # Cancer

    total_img_num = non_can_num + can_num

    print(
        "Number of Images of no cancer: {}".format(non_can_num)
    )  # images of Non cancer
    print("Number of Images of cancer : {}".format(can_num))  # images of cancer
    print("Total Number of Images : {}".format(total_img_num))

    data_insight_1 = pd.DataFrame(
        {"state of cancer": ["0", "1"], "Numbers of Patients": [non_can_num, can_num]}
    )
    bar = px.bar(
        data_frame=data_insight_1,
        x="state of cancer",
        y="Numbers of Patients",
        color="state of cancer",
    )
    bar.update_layout(
        title_text="Number of Patients with cancer (1) and patients with no cancer (0)",
        title_x=0.5,
    )
    # bar.show()

    non_img_arr = []
    can_img_arr = []
    print("Reading non_can_img")
    for img in non_can_img:
        n_img = cv2.imread(img, cv2.IMREAD_COLOR)
        n_img_size = cv2.resize(n_img, (50, 50), interpolation=cv2.INTER_LINEAR)
        non_img_arr.append([n_img_size, 0])

    print("Reading can_img")
    for img in can_img:
        c_img = cv2.imread(img, cv2.IMREAD_COLOR)
        c_img_size = cv2.resize(c_img, (50, 50), interpolation=cv2.INTER_LINEAR)
        can_img_arr.append([c_img_size, 1])

    X = []
    y = []

    breast_img_arr = np.concatenate((non_img_arr, can_img_arr))
    random.shuffle(breast_img_arr)

    for feature, label in breast_img_arr:
        X.append(feature)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("X shape : {}".format(X.shape))

    ####
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    return X_train, X_test, y_train, y_test


def average_params(weights_list):
    """
    Returns the average of the params.
    """
    # w_avg = copy.deepcopy(w[0])
    # for key in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[key] += w[i][key]
    #     w_avg[key] = torch.div(w_avg[key], len(w))

    w_avg = copy.deepcopy(weights_list[0])
    for weights in weights_list:
        w_avg += weights
    return w_avg / len(weights_list)


def train_client(global_round, model_path):
    num_channels = 1  # FIXME: get from db
    num_classes = 3  # FIXME: get from db
    use_gpu = False  # FIXME: get from db
    model = "cnn"  # FIXME: get from db
    local_ep = 1  # FIXME: get from db
    local_bs = 10  # FIXME: get from db

    start_time = timezone.now()

    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")  # run with cpu

    # load dataset
    X_train, X_test, y_train, y_test = get_dataset()

    # BUILD MODEL
    model = CNNModel()

    # Set the model to train and send it to device.
    global_params = read_params_from_s3(model_path)

    # Training
    train_loss_list, train_acc_list = [], []
    client = Device.objects.get(id=settings.CLIENT_ID)
    train_loss_list_str = client.train_loss_list
    if train_loss_list_str:
        train_loss_list = json.loads(train_loss_list_str)
    train_acc_list_str = client.train_acc_list
    if train_acc_list_str:
        train_acc_list = json.loads(train_acc_list_str)
    val_acc_list_str = client.val_acc_list
    if val_acc_list_str:
        val_acc_list = json.loads(val_acc_list_str)
    val_loss_list_str = client.val_loss_list
    if val_loss_list_str:
        val_loss_list = json.loads(val_loss_list_str)

    model.set_weights(global_params)
    history = model.train(X_train, y_train, X_test, y_test)
    # local_params, train_loss = local_update.update_weights(
    #     model=local_model, global_round=global_round
    # )
    client = Device.objects.get(id=settings.CLIENT_ID)
    # train loss
    train_loss = history["loss"]
    train_loss_list.extend(train_loss)
    client.train_loss = train_loss[-1]
    train_loss_list_str = json.dumps(train_loss_list)
    client.train_loss_list = train_loss_list_str
    # train accuracy
    train_acc = history["accuracy"]
    train_acc_list.extend(train_acc)
    client.train_acc = train_acc[-1]
    train_acc_list_str = json.dumps(train_acc_list)
    client.train_acc_list = train_acc_list_str
    # validation accuracy
    val_acc = history["val_accuracy"]
    val_acc_list.extend(val_acc)
    client.val_acc = val_acc[-1]
    val_acc_list_str = json.dumps(val_acc_list)
    client.val_acc_list = val_acc_list_str
    # validation loss
    val_loss = history["val_loss"]
    val_loss_list.extend(val_loss)
    client.val_loss = val_loss[-1]
    val_loss_list_str = json.dumps(val_loss_list)
    client.val_loss_list = val_loss_list_str

    client.current_global_round = global_round

    local_params = model.get_weights()
    buffer = io.BytesIO()
    pickle.dump(local_params, buffer)
    model_path = upload_params_to_s3(
        buffer.getvalue(),
        "client_1_params",
        "local_model_round_%s.pkl" % (global_round),
    )
    client.current_model_path = model_path
    client.save(
        update_fields=[
            "current_model_path",
            "current_global_round",
            "train_loss",
            "train_acc",
            "train_loss_list",
            "train_acc_list",
            "val_acc_list",
            "val_loss_list",
        ]
    )

    # STEP 6: Client call api to sends params to center
    print("STEP 6", timezone.now())
    res = requests.post(
        CLIENT_API_URL + "/client/params/sends",
        json={"global_round": global_round, "model_path": model_path},
    )
    print("/center/params/receives", res.json())
    print(
        f" \n Results after {local_ep} times of local training and {global_round} times global training:"
    )
    print("|---- Train Accuracy: {:.2f}%".format(100 * train_acc_list[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    plt.plot(train_acc_list)
    plt.plot(train_acc_list)  # FIXME: convert train_acc_list -> val_acc_list
    plt.title("Model Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("results_xray/model_acc_global_round_%s.png" % global_round)
    plt.close()

    print("\n Total Run Time: %s" % (timezone.now() - start_time))


def send_params_to_clients(global_round=None):
    if global_round is None:
        center = Device.objects.get(is_center=True)
        global_round = center.current_global_round


def train_center(global_round=None):
    """
    Params:
    - global_round: min = 1
    """
    if global_round > 1:
        # Show train loss, train acc and save them to db of the global training before
        clients_result = Device.objects.filter(is_center=False).values_list(
            "train_acc", "train_loss"
        )
        train_acc_list, train_loss_list = list(zip(*clients_result))
        avg_train_acc = sum(train_acc_list) / len(train_acc_list)
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        print(f" \nAvg Training Stats after {global_round - 1} global rounds:")
        print(f"Training Loss : {avg_train_loss}")
        print("Train Accuracy: %s \n" % avg_train_acc)
        center = Device.objects.get(is_center=True)
        center.train_acc = avg_train_acc
        center.train_loss = avg_train_loss
        center.save(update_fields=["train_acc", "train_loss"])

    num_channels = 1  # FIXME: get from db
    num_classes = 3  # FIXME: get from db
    epochs = 2  # FIXME: get from db
    use_gpu = False  # FIXME: get from db
    model = "cnn"  # FIXME: get from db

    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")  # run with cpu

    # BUILD MODEL
    model = CNNModel()

    # Training
    if global_round < epochs:
        train_loss_list, train_acc_list = [], []
        local_params_list, local_losses = [], []
        print(f"\n | Global Training Round : {global_round} |\n")

        if global_round == 1:
            # STEP 1: Center init params
            print("STEP 1", timezone.now())
            global_params = model.get_weights()
        else:
            # STEP 10: Center calculate params and retrain FL
            print("STEP 10", timezone.now())
            model_path_list = Device.objects.filter(
                is_center=False,
                current_global_round=global_round,
                current_model_path__isnull=False,
            ).values_list("model_path", flat=True)
            local_params_list = [read_params_from_s3(path) for path in model_path_list]
            global_params = average_params(local_params_list)

        buffer = io.BytesIO()
        pickle.dump(global_params, buffer)
        model_path = upload_params_to_s3(
            buffer.getvalue(),
            "center_params",
            "global_model_round_%s.pkl" % (global_round),
        )
        # STEP 2: Center create event and send params to clients
        print("STEP 2", timezone.now())
        res = requests.post(
            CENTER_API_URL + "/center/params/sends",
            json={"global_round": global_round, "model_path": model_path},
        )
        print("/center/params/sends", res)
