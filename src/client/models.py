from django.core.serializers.json import DjangoJSONEncoder
from django.db import models
from django.utils import timezone

# Create your models here.


class Client(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    name = models.CharField(max_length=64)
    current_params = models.JSONField(
        blank=True, null=True, default=dict, encoder=DjangoJSONEncoder)
