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
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from ..device.models import Device
from ..utils.aws_s3 import read_params_from_s3, upload_params_to_s3
from ..utils.storage import read_params_from_storage, upload_params_to_storage
from .models import CNNModel

# from os import listdir


def get_dataset_with_batch_size(batch_size):
    # create generator
    datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    # prepare an iterators for each dataset
    train_it = datagen.flow_from_directory(
        "src/train/dataset/train/",
        target_size=(224, 224),
        class_mode = 'categorical',
        batch_size=batch_size,
    )
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    train_it.mean = mean
    # confirm the iterator works
    batchX, batchy = train_it.next()
    print(
        "Batch shape=%s, min=%.3f, max=%.3f"
        % (batchX.shape, batchX.min(), batchX.max())
    )
    return train_it


def average_params(weights_list):
    """
    Returns the average of the params.
    """

    w_avg = np.array(weights_list[0])
    for weights in weights_list[1:]:
        w_avg += np.array(weights)
    return list(w_avg / len(weights_list))


def train_client(global_round, model_path):
    client = Device.objects.get(id=settings.CLIENT_ID)
    client.current_global_round = global_round
    client.save(update_fields=["current_global_round"])

    num_channels = client.num_channels
    num_classes = client.num_classes
    use_gpu = client.use_gpu
    local_ep = client.epochs
    local_bs = client.batch_size

    start_time = timezone.now()

    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")  # run with cpu

    # load dataset
    train_it = get_dataset_with_batch_size(local_bs)

    # BUILD MODEL
    model = CNNModel(num_channels=num_channels, num_classes=num_classes)

    # Set the model to train and send it to device.
    if settings.USE_AWS_STORAGE:
        global_params = read_params_from_s3(model_path)
    else:
        global_params = read_params_from_storage(model_path)

    # Training
    train_loss_list, train_acc_list, val_acc_list, val_loss_list = [], [], [], []
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
    history = model.train(train_it, local_ep, local_bs)
    print("-----Done training")

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
    if settings.USE_AWS_STORAGE:
        model_path = upload_params_to_s3(
            buffer.getvalue(),
            "client_1_params",
            "local_model_round_%s.pkl" % (global_round),
        )
    else:
        model_path = upload_params_to_storage(
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
            "val_acc",
            "val_loss",
            "train_loss_list",
            "train_acc_list",
            "val_acc_list",
            "val_loss_list",
        ]
    )

    # STEP 6: Client call api to sends params to center
    print("STEP 6", timezone.now())
    requests.post(
        client.api_url + "/client/params/sends",
        json={"global_round": global_round, "model_path": model_path},
    )
    print(
        f" \n Results after {local_ep} times of local training and {global_round} times global training:"
    )
    print("---- Train Accuracy: {:.2f}%".format(100 * train_acc_list[-1]))
    print("---- Validation Accuracy: {:.2f}%".format(100 * val_acc_list[-1]))
    print("---- Train Loss: {:.2f}".format(train_loss_list[-1]))
    print("---- Validation Loss: {:.2f}".format(val_loss_list[-1]))

    # accuracy
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.title("Model Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("src/train/results/client/model_acc_global_round_%s.png" % global_round)
    plt.close()

    # loss
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.title("Model Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(
        "src/train/results/client/model_loss_global_round_%s.png" % global_round
    )
    plt.close()


def send_params_to_clients(center, global_round=None):
    if global_round is None:
        global_round = center.current_global_round

    num_channels = center.num_channels
    num_classes = center.num_classes
    epochs = center.epochs
    use_gpu = center.use_gpu

    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")  # run with cpu

    # BUILD MODEL
    model = CNNModel(num_channels=num_channels, num_classes=num_classes)

    # Training
    if global_round <= epochs:
        local_params_list = []
        print(f"\n ----- Global Training Round : {global_round} |\n")

        if global_round == 1:
            # STEP 1: Center init params
            print("STEP 1", timezone.now())
            global_params = model.get_weights()
        else:
            # STEP 10: Center calculate params and retrain FL
            print("STEP 10", timezone.now())
            model_path_list = Device.objects.filter(
                is_center=False,
                current_global_round=global_round - 1,
                current_model_path__isnull=False,
            ).values_list("current_model_path", flat=True)
            if settings.USE_AWS_STORAGE:
                local_params_list = [
                    read_params_from_s3(path) for path in model_path_list
                ]
            else:
                local_params_list = [
                    read_params_from_storage(path) for path in model_path_list
                ]
            global_params = average_params(local_params_list)

        buffer = io.BytesIO()
        pickle.dump(global_params, buffer)
        if settings.USE_AWS_STORAGE:
            model_path = upload_params_to_s3(
                buffer.getvalue(),
                "center_params",
                "global_model_round_%s.pkl" % (global_round),
            )
        else:
            model_path = upload_params_to_storage(
                buffer.getvalue(),
                "center_params",
                "global_model_round_%s.pkl" % (global_round),
            )
        # STEP 2: Center create event and send params to clients
        print("STEP 2", timezone.now())
        requests.post(
            center.api_url + "/center/params/sends",
            json={"global_round": global_round, "model_path": model_path},
        )


def train_center(global_round):
    """
    Params:
    - global_round: min = 1
    """
    center = Device.objects.get(is_center=True)
    if global_round > 1:
        # Show train loss, train acc and save them to db of the global training before
        clients_result = Device.objects.filter(is_center=False).values_list(
            "train_acc", "train_loss", "val_acc", "val_loss"
        )
        (
            local_train_acc_list,
            local_train_loss_list,
            local_val_acc_list,
            local_val_loss_list,
        ) = list(zip(*clients_result))
        avg_train_acc = sum(local_train_acc_list) / len(local_train_acc_list)
        avg_train_loss = sum(local_train_loss_list) / len(local_train_loss_list)
        avg_val_acc = sum(local_val_acc_list) / len(local_val_acc_list)
        avg_val_loss = sum(local_val_loss_list) / len(local_val_loss_list)
        print(f" \nAvg Training Stats after {global_round - 1} global rounds:")
        print(f"----- Training Loss : {avg_train_loss}")
        print("---- Train Accuracy: {:.2f}% \n".format(100 * avg_train_acc))
        center.current_global_round = global_round

        train_acc_list, train_loss_list, val_acc_list, val_loss_list = [], [], [], []
        # train acc
        train_acc_list_str = center.train_acc_list
        if train_acc_list_str:
            train_acc_list = json.loads(train_acc_list_str)
        train_acc_list.append(avg_train_acc)
        train_acc_list_str = json.dumps(train_acc_list)
        center.train_acc_list = train_acc_list_str
        center.train_acc = avg_train_acc
        # train loss
        train_loss_list_str = center.train_loss_list
        if train_loss_list_str:
            train_loss_list = json.loads(train_loss_list_str)
        train_loss_list.append(avg_train_loss)
        train_loss_list_str = json.dumps(train_loss_list)
        center.train_loss_list = train_loss_list_str
        center.train_loss = avg_train_loss
        # validation accuracy
        val_acc_list_str = center.val_acc_list
        if val_acc_list_str:
            val_acc_list = json.loads(val_acc_list_str)
        val_acc_list.append(avg_val_acc)
        val_acc_list_str = json.dumps(val_acc_list)
        center.val_acc_list = val_acc_list_str
        center.val_acc = avg_val_acc
        # validation loss
        val_loss_list_str = center.val_loss_list
        if val_loss_list_str:
            val_loss_list = json.loads(val_loss_list_str)
        val_loss_list.append(avg_val_loss)
        val_loss_list_str = json.dumps(val_loss_list)
        center.val_loss_list = val_loss_list_str
        center.val_loss = avg_val_loss

        center.save(
            update_fields=[
                "train_acc",
                "train_loss",
                "val_acc",
                "val_loss",
                "current_global_round",
                "train_loss_list",
                "train_acc_list",
                "val_acc_list",
                "val_loss_list",
            ]
        )

    send_params_to_clients(center, global_round)
