import copy
import io
import json
import pickle
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from django.conf import settings
from django.utils import timezone
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

from .. import CLIENT_API_URL, ROOT_PATH
from ..device.models import Device
from ..utils.aws_s3 import read_params_from_s3, upload_params_to_s3
from .datasets import CustomDataset
from .models import CNN, CNNOpt
from .update import LocalUpdate, test_inference


def get_dataset():
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    train_img_dir = f"{ROOT_PATH}/data/custom/train/images"
    train_annotations_file = f"{ROOT_PATH}/data/custom/train/annotations.csv"
    test_img_dir = f"{ROOT_PATH}/data/custom/test/images"
    test_annotations_file = f"{ROOT_PATH}/data/custom/test/annotations.csv"
    # train_img_dir = f"src/client_training/data/custom/train/images"
    # train_annotations_file = f"src/client_training/data/custom/train/annotations.csv"
    # test_img_dir = f"src/client_training/data/custom/test/images"
    # test_annotations_file = f"src/client_training/data/custom/test/annotations.csv"

    train_dataset = CustomDataset(
        annotations_file=train_annotations_file,
        img_dir=train_img_dir,
    )
    test_dataset = CustomDataset(
        annotations_file=test_annotations_file,
        img_dir=test_img_dir,
    )

    return train_dataset, test_dataset


def average_params(w):
    """
    Returns the average of the params.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details():
    global_round = 10  # FIXME: get from db
    model = "cnn"  # FIXME: get from db
    optimizer = "sgd"  # FIXME: get from db
    lr = 0.01  # FIXME: get from db
    local_bs = 10  # FIXME: get from db
    local_ep = 1  # FIXME: get from db

    print("\nExperimental details:")
    print(f"    Model     : {model}")
    print(f"    Optimizer : {optimizer}")
    print(f"    Learning  : {lr}")
    print(f"    Global Round   : {global_round}\n")

    print("    Federated parameters:")
    print(f"    Local Batch size   : {local_bs}")
    print(f"    Local Epochs       : {local_ep}\n")
    return


def train_client(global_round, model_path):
    num_channels = 1  # FIXME: get from db
    num_classes = 3  # FIXME: get from db
    use_gpu = False  # FIXME: get from db
    model = "cnn"  # FIXME: get from db
    local_ep = 1  # FIXME: get from db
    local_bs = 10  # FIXME: get from db

    start_time = time.time()

    logger = SummaryWriter(f"{ROOT_PATH}/results/client/logs")

    exp_details()

    if use_gpu:
        torch.cuda.set_device(use_gpu)
    device = "cuda" if use_gpu else "cpu"
    print(device)

    # load dataset
    train_dataset, test_dataset = get_dataset()

    # BUILD MODEL
    if model == "cnn":
        # Convolutional neural network
        local_model = CNN(num_channels=num_channels, num_classes=num_classes)
    elif model == "cnnopt":
        local_model = CNNOpt()
    else:
        exit("Error: unrecognized model")

    # Set the model to train and send it to device.
    local_model.to(device)
    local_model.train()

    # TODO: get params of center
    # global_params = local_model.state_dict()
    global_params = read_params_from_s3(model_path)

    # Training
    train_loss_list, train_acc_list = [], []
    client = Device.objects.get(id=settings.CLIENT_ID)
    train_loss_list_str = client.train_loss_list
    if train_loss_list_str:
        train_loss_list = json.loads(train_loss_list_str)
    train_acc_list_str = client.train_acc_list
    if train_loss_list_str:
        train_acc_list = json.loads(train_acc_list_str)

    local_update = LocalUpdate(dataset=train_dataset, logger=logger)
    local_model.load_state_dict(global_params, strict=True)
    local_params, train_loss = local_update.update_weights(
        model=local_model, global_round=global_round
    )
    print("loss_1:", train_loss)
    client = Device.objects.get(id=settings.CLIENT_ID)
    # train loss
    train_loss_list.append(train_loss)
    client.current_loss = train_loss
    client.train_loss_list = train_loss_list
    train_loss_list_str = json.dumps(train_loss_list)
    # train acc
    train_acc, loss = local_model.inference(model=local_model)
    print("loss_2:", loss)
    train_acc_list.append(train_acc)
    client.current_loss = train_acc
    client.train_acc_list = train_acc_list
    train_acc_list_str = json.dumps(train_acc_list)
    # _, pred_labels = torch.max(outputs, 1)
    # pred_labels = pred_labels.view(-1)
    # correct += torch.sum(torch.eq(pred_labels, labels)).item()
    # total += len(labels)

    client.current_global_round = global_round

    buffer = io.BytesIO()
    pickle.dump(local_params, buffer)
    model_path = upload_params_to_s3(
        buffer.getvalue(), "client_1_params", "local_model_round_%s.pkl" % (global_round))
    # TODO: update current_model_path of client on db
    client.current_model_path = model_path
    client.save(update_fields=["current_model_path", "current_global_round",
                "current_loss", "train_loss_list", "train_acc_list"])

    # STEP 6: Client call api to sends params to center
    print("STEP 6", timezone.now())
    res = requests.post(
        CLIENT_API_URL + "/client/params/sends", json={"global_round": global_round, "model_path": model_path}
    )
    print("/center/params/receives", res.json())
    # Test inference after completion of training
    test_acc, test_loss = test_inference(local_model, test_dataset)

    print(
        f" \n Results after {local_ep} times of local training and {global_round} times global training:")
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_acc_list[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss_list and train_acc_list:
    file_name = "{}/results/client/{}_{}_E[{}]_B[{}].pkl".format(
        ROOT_PATH,
        model,
        global_round,
        local_ep,
        local_bs,
    )

    with open(file_name, "wb") as f:
        pickle.dump([train_loss_list, train_acc_list], f)

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

    # PLOTTING
    matplotlib.use("Agg")

    # Plot Loss curve
    plt.figure()
    plt.title("Training Loss vs Communication rounds")
    plt.plot(range(len(train_loss_list)), train_loss_list, color="r")
    plt.ylabel("Training loss")
    plt.xlabel("Communication Rounds")
    plt.savefig(
        "{}/results/client[{}]/fed_{}_{}_E[{}]_B[{}]_loss.png".format(
            ROOT_PATH,
            settings.CLIENT_ID,
            model,
            global_round,
            local_ep,
            local_bs,
        )
    )

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title("Average Accuracy vs Communication rounds")
    plt.plot(range(len(train_acc_list)), train_acc_list, color="k")
    plt.ylabel("Average Accuracy")
    plt.xlabel("Communication Rounds")
    plt.savefig(
        "{}/results/client[{}]/fed_{}_{}_E[{}]_B[{}]_acc.png".format(
            ROOT_PATH,
            settings.CLIENT_ID,
            model,
            global_round,
            local_ep,
            local_bs,
        )
    )
