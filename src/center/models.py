from django.core.serializers.json import DjangoJSONEncoder
from django.db import models
from django.utils import timezone

from . import EventType
from ..client.models import Client

# Create your models here.


class CenterEvent(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    global_round = models.IntegerField()
    event_type = models.CharField(max_length=128, choices=EventType.choices)
    client = models.ForeignKey(
        Client, related_name="events", null=True, blank=True, on_delete=models.SET_NULL)
    model_path = models.CharField(
        max_length=512, null=True, blank=True)

    class Meta:
        db_table = "center_event"


class CenterConfig(models.Model):
    name = models.CharField(max_length=128)
    value = models.TextField()
    data_type = models.CharField(max_length=64)

    class Meta:
        db_table = "center_config"
