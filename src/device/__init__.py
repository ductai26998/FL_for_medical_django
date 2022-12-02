from django.db.models import TextChoices


class EventType(TextChoices):
    CENTER_RECEIVED_PARAMS = "center_received_params"
    CENTER_SENT_PARAMS = "center_sent_params"
