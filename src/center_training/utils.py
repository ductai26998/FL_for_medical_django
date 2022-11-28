import copy
import io
import json
import os
import pickle
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from ..utils.aws_s3 import read_params_from_s3, upload_params_to_s3
from . import API_URL, ROOT_PATH
from .datasets import CustomDataset
from .models import CNN, CNNOpt
from .update import LocalUpdate, test_inference


def get_dataset():
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    num_users = 2  # FIXME: get from db
    iid = True  # FIXME: get from db
    unequal = False  # FIXME: get from db

    train_img_dir = f"{ROOT_PATH}/data/custom/train/images"
    train_annotations_file = f"{ROOT_PATH}/data/custom/train/annotations.csv"
    test_img_dir = f"{ROOT_PATH}/data/custom/test/images"
    test_annotations_file = f"{ROOT_PATH}/data/custom/test/annotations.csv"
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
    epochs = 10  # FIXME: get from db
    model = "cnn"  # FIXME: get from db
    optimizer = "sgd"  # FIXME: get from db
    lr = 0.01  # FIXME: get from db
    iid = True  # FIXME: get from db
    frac = 0.1  # FIXME: get from db
    local_bs = 10  # FIXME: get from db
    local_ep = 100  # FIXME: get from db

    print("\nExperimental details:")
    print(f"    Model     : {model}")
    print(f"    Optimizer : {optimizer}")
    print(f"    Learning  : {lr}")
    print(f"    Global Rounds   : {epochs}\n")

    print("    Federated parameters:")
    if iid:
        print("    IID")
    else:
        print("    Non-IID")
    print(f"    Fraction of users  : {frac}")
    print(f"    Local Batch size   : {local_bs}")
    print(f"    Local Epochs       : {local_ep}\n")
    return


def train_center():
    num_channels = 1  # FIXME: get from db
    num_classes = 3  # FIXME: get from db
    num_users = 2  # FIXME: get from db
    epochs = 2  # FIXME: get from db
    use_gpu = False  # FIXME: get from db
    model = "cnn"  # FIXME: get from db
    dataset = "custom"  # FIXME: get from db
    iid = True  # FIXME: get from db
    frac = 0.1  # FIXME: get from db
    local_ep = 100  # FIXME: get from db
    local_bs = 10  # FIXME: get from db

    start_time = time.time()

    # define paths
    logger = SummaryWriter(f"{ROOT_PATH}/results/logs")

    exp_details()

    if use_gpu:
        torch.cuda.set_device(use_gpu)
    device = "cuda" if use_gpu else "cpu"
    print(device)

    # load dataset and user groups
    train_dataset, test_dataset = get_dataset()

    # BUILD MODEL
    if model == "cnn":
        # Convolutional neural network
        global_model = CNN(num_channels=num_channels, num_classes=num_classes)
    elif model == "cnnopt":
        global_model = CNNOpt()
    else:
        exit("Error: unrecognized model")

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy params
    global_params = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    print_every = 2

    for epoch in tqdm(range(epochs)):
        local_params_list, local_losses = [], []
        print(f"\n | Global Training Round : {epoch+1} |\n")

        global_model.train()
        params = global_model.state_dict()
        buffer = io.BytesIO()
        pickle.dump(params, buffer)
        model_path = upload_params_to_s3(
            buffer.getvalue(), "center_params", "global_model_round_%s.pkl" % (epoch+1))
        res = requests.post(
            API_URL + "/center/params/sends", json={"global_round": epoch + 1, "model_path": model_path}
        )
        print("response_1", res.json())

        # update global params
        # TODO: receive params and update here
        # response = requests.post(
        #     API_URL + "/center/params/receives", json={"client_id": 1, "global_round": epoch + 1, "params": params}
        # )
        # print("response_2", response.json())
        response = requests.get(
            API_URL + "/center/params", json={"global_round": epoch + 1}
        )
        response = response.json()
        if "data" in response:
            model_paths = response["data"]
            local_params_list = [read_params_from_s3(
                path) for path in model_paths]
        print(local_params_list)
        global_params = average_params(local_params_list)

        # update global params
        global_model.load_state_dict(global_params)

        # loss_avg = sum(local_losses) / len(local_losses)
        # train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(num_users):
            local_model = LocalUpdate(
                dataset=train_dataset, logger=logger
            )
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Loss : {np.mean(np.array(train_loss))}")
            print("Train Accuracy: {:.2f}% \n".format(
                100 * train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(global_model, test_dataset)

    print(f" \n Results after {epochs} global rounds of training:")
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = "{}/results/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl".format(
        ROOT_PATH,
        dataset,
        model,
        epochs,
        frac,
        iid,
        local_ep,
        local_bs,
    )

    with open(file_name, "wb") as f:
        pickle.dump([train_loss, train_accuracy], f)

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

    # PLOTTING

    matplotlib.use("Agg")

    # Plot Loss curve
    plt.figure()
    plt.title("Training Loss vs Communication rounds")
    plt.plot(range(len(train_loss)), train_loss, color="r")
    plt.ylabel("Training loss")
    plt.xlabel("Communication Rounds")
    plt.savefig(
        "{}/results/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png".format(
            ROOT_PATH,
            dataset,
            model,
            epochs,
            frac,
            iid,
            local_ep,
            local_bs,
        )
    )

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title("Average Accuracy vs Communication rounds")
    plt.plot(range(len(train_accuracy)), train_accuracy, color="k")
    plt.ylabel("Average Accuracy")
    plt.xlabel("Communication Rounds")
    plt.savefig(
        "{}/results/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png".format(
            ROOT_PATH,
            dataset,
            model,
            epochs,
            frac,
            iid,
            local_ep,
            local_bs,
        )
    )
