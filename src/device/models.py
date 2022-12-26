from django.db import models
from django.utils import timezone

# Create your models here.


class DeviceManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_active=True)


class Device(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    name = models.CharField(max_length=64)
    is_center = models.BooleanField(default=False)
    current_global_round = models.IntegerField(null=True, blank=True)
    current_model_path = models.CharField(
        max_length=512, null=True, blank=True)
    train_acc = models.FloatField(null=True, blank=True)
    train_loss = models.FloatField(null=True, blank=True)
    train_acc_list = models.TextField(null=True, blank=True)
    train_loss_list = models.TextField(null=True, blank=True)
    val_acc = models.FloatField(null=True, blank=True)
    val_loss = models.FloatField(null=True, blank=True)
    val_acc_list = models.TextField(null=True, blank=True)
    val_loss_list = models.TextField(null=True, blank=True)
    api_url = models.CharField(max_length=256)
    is_active = models.BooleanField(default=True)

    # CONFIG
    epochs = models.IntegerField(null=True, blank=True)
    use_gpu = models.BooleanField(default=False)
    # numbers of image layer when processing
    num_channels = models.IntegerField(null=True, blank=True)
    # total class of dataset
    num_classes = models.IntegerField(null=True, blank=True)
    batch_size = models.IntegerField(null=True, blank=True)
    dataset_size = models.IntegerField(null=True, blank=True)

    objects = DeviceManager()

    class Meta:
        db_table = "device"

class EventManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(device__is_active=True)

class Event(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)
    event_type = models.CharField(max_length=128)
    device = models.ForeignKey(
        Device, on_delete=models.CASCADE, related_name="events")
    model_path = models.CharField(max_length=512, null=True, blank=True)
    global_round = models.IntegerField(null=True, blank=True)
    status = models.CharField(max_length=128, null=True, blank=True)
    train_acc = models.FloatField(null=True, blank=True)
    train_loss = models.FloatField(null=True, blank=True)

    objects = EventManager()

    class Meta:
        db_table = "event"
