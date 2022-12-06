from django.db.models import TextChoices


class EventType(TextChoices):
    CENTER_RECEIVED_PARAMS = "center_received_params"
    CENTER_SENT_PARAMS = "center_sent_params"


class ErrorCode:
    EXISTED = "existed"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    PROCESSING_ERROR = "processing_error"
    REQUIRED = "required"


class EventStatus:
    STARTED = "started"
    FAILED = "failed"
    COMPLETED = "completed"
