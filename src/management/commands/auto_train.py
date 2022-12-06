import json

import matplotlib.pyplot as plt
from django.core.management.base import BaseCommand
from django.utils import timezone

from ...device.models import Device
from ...train.utils import send_params_to_clients


class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        start_time = timezone.now()
        center = Device.objects.get(is_center=True)
        send_params_to_clients(center=center)
        # Show train loss, train acc and save them to db of the last global training
        clients_result = Device.objects.filter(is_center=False).values_list(
            "train_acc", "train_loss", "val_acc", "val_loss"
        )
        print("clients_result:", clients_result)
        if clients_result:
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

            center = Device.objects.get(is_center=True)
            center.train_acc = avg_train_acc
            center.train_loss = avg_train_loss

            train_acc_list, train_loss_list, val_acc_list, val_loss_list = (
                [],
                [],
                [],
                [],
            )
            # train acc
            train_acc_list_str = center.train_acc_list
            if train_acc_list_str:
                train_acc_list = json.loads(train_acc_list_str)
            train_acc_list.append(avg_train_acc)
            train_acc_list_str = json.dumps(train_acc_list)
            center.train_acc_list = train_acc_list_str
            # train loss
            train_loss_list_str = center.train_loss_list
            if train_loss_list_str:
                train_loss_list = json.loads(train_loss_list_str)
            
            train_loss_list_str = json.dumps(train_loss_list)
            center.train_loss_list = train_loss_list_str
            # validation accuracy
            val_acc_list_str = center.val_acc_list
            if val_acc_list_str:
                val_acc_list = json.loads(val_acc_list_str)
            val_acc_list.append(avg_val_acc)
            val_acc_list_str = json.dumps(val_acc_list)
            center.val_acc_list = val_acc_list_str
            # validation loss
            val_loss_list_str = center.val_loss_list
            if val_loss_list_str:
                val_loss_list = json.loads(val_loss_list_str)
            val_loss_list.append(avg_val_loss)
            val_loss_list_str = json.dumps(val_loss_list)
            center.val_loss_list = val_loss_list_str

            center.save(
                update_fields=[
                    "train_acc",
                    "train_loss",
                    "train_acc_list",
                    "train_loss_list",
                    "val_acc_list",
                    "val_loss_list",
                ]
            )

            end_time = timezone.now()
            print("\n Total Run Time: %s" % (end_time - start_time))
            print(f" \n Results after {center.epochs} global rounds of training:")
            print("|---- Train Accuracy: {:.2f}%".format(100 * avg_train_acc))
            print("|---- Train Loss: {:.2f}%".format(100 * avg_train_loss))

            time_str = end_time.strftime("%d-%m-%Y_%H-%M-%S")
            # accuracy
            print("train_acc_list", train_acc_list)
            plt.plot(train_acc_list)
            plt.plot(val_acc_list)
            plt.title("Model Accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train", "validation"], loc="upper left")
            plt.savefig(
                "src/train/results/center/train_acc_E[%s]_%s.png"
                % (center.epochs, time_str)
            )
            plt.close()

            # loss
            plt.plot(train_loss_list)
            plt.plot(val_loss_list)
            plt.title("Model Loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "validation"], loc="upper left")
            plt.savefig(
                "src/train/results/center/train_loss_E[%s]_%s.png"
                % (center.epochs, time_str)
            )
            plt.close()
