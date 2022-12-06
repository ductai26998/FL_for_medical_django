from django.core.management.base import BaseCommand
from django.utils import timezone

from ...device.models import Device
from ...train.utils import send_params_to_clients
import matplotlib.pyplot as plt


class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        start_time = timezone.now()
        center = Device.objects.get(is_center=True)
        send_params_to_clients(center=center)
        # Show train loss, train acc and save them to db of the last global training
        clients_result = Device.objects.filter(is_center=False).values_list(
            "train_acc", "train_loss"
        )
        print("clients_result:", clients_result)
        if clients_result:
            train_acc_list, train_loss_list = list(zip(*clients_result))
            avg_train_acc = sum(train_acc_list) / len(train_acc_list)
            avg_train_loss = sum(train_loss_list) / len(train_loss_list)
            center = Device.objects.get(is_center=True)
            center.train_acc = avg_train_acc
            center.train_loss = avg_train_loss
            center.save(update_fields=["train_acc", "train_loss"])
            end_time = timezone.now()
            print("\n Total Run Time: %s" % (end_time - start_time))
            print(f" \n Results after {center.epochs} global rounds of training:")
            print("|---- Train Accuracy: {:.2f}%".format(100 * avg_train_acc))
            print("|---- Train Loss: {:.2f}%".format(100 * avg_train_loss))

            time_str = end_time.strftime("%d-%m-%Y_%H-%M-%S")
            # accuracy
            train_acc_list = train_acc_list.append(avg_train_acc)
            plt.plot(train_acc_list)
            plt.title("Model Accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train"], loc="upper left")
            plt.savefig(
                "src/train/results/center/train_acc_E[%s]_%s.png"
                % (center.epochs, time_str)
            )
            plt.close()

            # loss
            train_loss_list = train_loss_list.append(avg_train_loss)
            plt.plot(train_loss_list)
            plt.title("Model Loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train"], loc="upper left")
            plt.savefig(
                "src/train/results/center/train_loss_E[%s]_%s.png"
                % (center.epochs, time_str)
            )
            plt.close()
