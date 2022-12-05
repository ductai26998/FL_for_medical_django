from django.core.management.base import BaseCommand
from django.utils import timezone

from ...device.models import Device
from ...train.utils import send_params_to_clients


class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        start_time = timezone.now()
        send_params_to_clients()
        center = Device.objects.get(is_center=True)
        clients_result = Device.objects.filter(is_center=False).values_list(
            "train_acc", "train_loss"
        )
        train_acc_list, train_loss_list = list(zip(*clients_result))
        avg_train_acc = sum(train_acc_list) / len(train_acc_list)
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        center = Device.objects.get(is_center=True)
        center.train_acc = avg_train_acc
        center.train_loss = avg_train_loss
        center.save(update_fields=["train_acc", "train_loss"])
        print("\n Total Run Time: {0:0.4f}".format(timezone.now() - start_time))
        print(f" \n Results after {center.epochs} global rounds of training:")
        print("|---- Train Accuracy: {:.2f}%".format(100 * center.avg_train_acc))
        print("|---- Train Loss: {:.2f}%".format(100 * center.avg_train_loss))
