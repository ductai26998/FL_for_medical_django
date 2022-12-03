import copy
import io
import pickle
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from django.utils import timezone
from tensorboardX import SummaryWriter

from .. import CENTER_API_URL, ROOT_PATH
from ..utils.aws_s3 import read_params_from_s3, upload_params_to_s3
from .models import CNN, CNNOpt


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

    print("\nExperimental details:")
    print(f"    Model     : {model}")
    print(f"    Optimizer : {optimizer}")
    print(f"    Global Rounds   : {epochs}\n")

    print("    Federated parameters:")
    return


def train_center(global_round=1):
    num_channels = 1  # FIXME: get from db
    num_classes = 3  # FIXME: get from db
    epochs = 2  # FIXME: get from db
    use_gpu = False  # FIXME: get from db
    model = "cnn"  # FIXME: get from db

    start_time = time.time()

    logger = SummaryWriter(f"{ROOT_PATH}/results/client/logs")

    exp_details()

    if use_gpu:
        torch.cuda.set_device(use_gpu)
    device = "cuda" if use_gpu else "cpu"
    print(device)

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

    # Training
    print_every = 2

    if global_round < epochs:
        train_loss_list, train_acc_list = [], []
        local_params_list, local_losses = [], []
        print(f"\n | Global Training Round : {global_round} |\n")

        global_model.train()
        if global_round == 1:
            # STEP 1: Center init params
            print("STEP 1", timezone.now())
            global_params = global_model.state_dict()
        else:
            # STEP 10: Center calculate params and retrain FL
            print("STEP 10", timezone.now())
            response = requests.get(
                CENTER_API_URL + "/center/get-client-params", json={"global_round": global_round}
            )
            print("/center/get-client-params", response.json())
            response = response.json()
            if "data" in response:
                model_paths = response["data"]
                local_params_list = [read_params_from_s3(
                    path) for path in model_paths]
            global_params = average_params(local_params_list)

        buffer = io.BytesIO()
        pickle.dump(global_params, buffer)
        model_path = upload_params_to_s3(
            buffer.getvalue(), "center_params", "global_model_round_%s.pkl" % (global_round))
        # STEP 2: Center create event and send params to clients
        print("STEP 2", timezone.now())
        res = requests.post(
            CENTER_API_URL + "/center/params/sends", json={"global_round": global_round, "model_path": model_path}
        )
        print("/center/params/sends", res.json())

        # loss_avg = sum(local_losses) / len(local_losses)
        # train_loss.append(loss_avg)

        # # Calculate avg training accuracy over all users at every epoch
        # list_acc, list_loss = [], []
        # global_model.eval()
        # for c in range(num_users):
        #     local_update = LocalUpdate(
        #         dataset=train_dataset, logger=logger
        #     )
        #     acc, loss = local_update.inference(model=global_model)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (global_round) % print_every == 0:
            print(f" \nAvg Training Stats after {global_round} global rounds:")
            print(f"Training Loss : {np.mean(np.array(train_loss))}")
            # print("Train Accuracy: {:.2f}% \n".format(
            #     100 * train_accuracy[-1]))

    print(f" \n Results after {epochs} global rounds of training:")
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = "{}/results/client/{}_{}.pkl".format(
        ROOT_PATH,
        model,
        epochs,
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
        "{}/results/client/fed_{}_{}_loss.png".format(
            ROOT_PATH,
            model,
            epochs,
        )
    )

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title("Average Accuracy vs Communication rounds")
    plt.plot(range(len(train_accuracy)), train_accuracy, color="k")
    plt.ylabel("Average Accuracy")
    plt.xlabel("Communication Rounds")
    plt.savefig(
        "{}/results/fed_{}_{}_acc.png".format(
            ROOT_PATH,
            model,
            epochs,
        )
    )
